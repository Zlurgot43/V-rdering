from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import requests
import math
import os
import json
import tempfile
import anthropic
import pdfplumber

app = FastAPI(title="Värderingsmotor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FMP_KEY = os.environ.get("FMP_API_KEY", "demo")
FMP_BASE = "https://financialmodelingprep.com/stable"
_claude_client = None


def get_claude():
    global _claude_client
    if _claude_client is None:
        _claude_client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
    return _claude_client


# ── HELPERS ───────────────────────────────────────────────────────────────────

def fmp(path: str, extra: dict = {}) -> dict | list:
    params = {"apikey": FMP_KEY, **extra}
    r = requests.get(f"{FMP_BASE}{path}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def safe(val, default=None):
    try:
        if val is None:
            return default
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def signal(upside):
    if upside is None:
        return "OKÄND"
    return "KÖPVÄRD" if upside > 15 else "NEUTRAL" if upside > -10 else "DYR"


def upside_pct(fair, price):
    if fair and price:
        return round(((fair - price) / price) * 100, 1)
    return None


# ── DATAHÄMTNING (FMP) ────────────────────────────────────────────────────────

def fetch_data(ticker: str) -> dict:
    ticker = ticker.upper()
    quote_data = fmp("/quote", {"symbol": ticker})
    if not quote_data or isinstance(quote_data, dict):
        raise ValueError(f"Ingen data för {ticker}")
    q = quote_data[0] if isinstance(quote_data, list) else quote_data

    try:
        profile_data = fmp("/profile", {"symbol": ticker})
        profile = profile_data[0] if isinstance(profile_data, list) and profile_data else {}
    except Exception:
        profile = {}
    try:
        cf_data = fmp("/cash-flow-statement", {"symbol": ticker, "limit": 1})
        cf = cf_data[0] if isinstance(cf_data, list) and cf_data else {}
    except Exception:
        cf = {}
    try:
        bs_data = fmp("/balance-sheet-statement", {"symbol": ticker, "limit": 1})
        bs = bs_data[0] if isinstance(bs_data, list) and bs_data else {}
    except Exception:
        bs = {}
    try:
        is_data = fmp("/income-statement", {"symbol": ticker, "limit": 1})
        inc = is_data[0] if isinstance(is_data, list) and is_data else {}
    except Exception:
        inc = {}
    try:
        ratios_data = fmp("/ratios-ttm", {"symbol": ticker})
        ratios = ratios_data[0] if isinstance(ratios_data, list) and ratios_data else {}
    except Exception:
        ratios = {}

    price      = safe(q.get("price"))
    market_cap = safe(q.get("marketCap"))
    shares     = safe(q.get("sharesOutstanding"))
    eps        = safe(q.get("eps"))
    book_value_total = safe(bs.get("totalStockholdersEquity"))
    book_value = (book_value_total / shares) if (book_value_total and shares and shares > 0) else safe(ratios.get("bookValuePerShareTTM"))
    total_debt = safe(bs.get("totalDebt"))
    cash       = safe(bs.get("cashAndCashEquivalents"))
    ebitda     = safe(inc.get("ebitda")) or safe(q.get("ebitda"))
    revenue    = safe(inc.get("revenue"))
    net_income = safe(inc.get("netIncome"))
    fcf        = safe(cf.get("freeCashFlow"))
    op_cf      = safe(cf.get("operatingCashFlow"))
    enterprise_value = None
    if market_cap and total_debt and cash:
        enterprise_value = market_cap + total_debt - cash
    elif market_cap:
        enterprise_value = market_cap

    return dict(
        ticker=ticker, name=profile.get("companyName") or q.get("name") or ticker,
        sector=profile.get("sector") or "Okänd",
        industry=profile.get("industry") or "Okänd",
        currency=profile.get("currency") or "USD",
        country=profile.get("country") or "",
        price=price, market_cap=market_cap, enterprise_value=enterprise_value,
        eps=eps, revenue=revenue, net_income=net_income, ebitda=ebitda,
        book_value=book_value, total_debt=total_debt, cash=cash, shares=shares,
        fcf=fcf, op_cf=op_cf,
        beta=safe(profile.get("beta"), 1.0),
        growth_5y=safe(ratios.get("revenueGrowthTTM"), 0.08),
        forward_pe=safe(ratios.get("priceEarningsRatioTTM")), peg=None,
    )


# ── SEKTORSSNITT ──────────────────────────────────────────────────────────────

SECTOR_PE = {"Technology": 28, "Healthcare": 22, "Financial Services": 14,
             "Consumer Cyclical": 20, "Consumer Defensive": 18, "Energy": 12,
             "Utilities": 16, "Industrials": 18, "Basic Materials": 14,
             "Real Estate": 30, "Communication Services": 22}
SECTOR_PB = {"Technology": 6, "Healthcare": 4, "Financial Services": 1.3,
             "Consumer Cyclical": 3, "Consumer Defensive": 4, "Energy": 1.5,
             "Utilities": 1.5, "Industrials": 2.5, "Basic Materials": 1.8,
             "Real Estate": 1.4, "Communication Services": 3}
SECTOR_EV = {"Technology": 20, "Healthcare": 15, "Financial Services": 10,
             "Consumer Cyclical": 12, "Consumer Defensive": 13, "Energy": 7,
             "Utilities": 10, "Industrials": 12, "Basic Materials": 9,
             "Real Estate": 18, "Communication Services": 15}


# ── VÄRDERINGSMODELLER ────────────────────────────────────────────────────────

def model_pe(d):
    if not d["eps"] or not d["price"] or d["eps"] <= 0:
        return {"available": False, "reason": "EPS saknas eller negativt"}
    bm = SECTOR_PE.get(d["sector"], 18)
    fv = d["eps"] * bm
    up = upside_pct(fv, d["price"])
    return {"available": True, "model": "P/E-värdering",
            "current_pe": round(d["price"] / d["eps"], 1), "sector_benchmark_pe": bm,
            "fair_value": round(fv, 2), "current_price": d["price"],
            "upside_pct": up, "signal": signal(up),
            "note": f"Sektorssnitt ({d['sector']}): {bm}x"}


def model_pb(d):
    if not d["book_value"] or not d["price"] or d["book_value"] <= 0:
        return {"available": False, "reason": "Bokfört värde saknas"}
    bm = SECTOR_PB.get(d["sector"], 2.5)
    fv = d["book_value"] * bm
    up = upside_pct(fv, d["price"])
    return {"available": True, "model": "P/B-värdering",
            "current_pb": round(d["price"] / d["book_value"], 2), "sector_benchmark_pb": bm,
            "book_value_per_share": round(d["book_value"], 2),
            "fair_value": round(fv, 2), "current_price": d["price"],
            "upside_pct": up, "signal": signal(up)}


def model_ev_ebitda(d):
    if not d["ebitda"] or not d["enterprise_value"] or not d["shares"] or d["ebitda"] <= 0:
        return {"available": False, "reason": "EBITDA eller EV saknas"}
    bm = SECTOR_EV.get(d["sector"], 12)
    fair_ev = d["ebitda"] * bm
    eq = fair_ev - (d["total_debt"] or 0) + (d["cash"] or 0)
    fv = eq / d["shares"]
    up = upside_pct(fv, d["price"])
    return {"available": True, "model": "EV/EBITDA-värdering",
            "current_ev_ebitda": round(d["enterprise_value"] / d["ebitda"], 1),
            "sector_benchmark": bm, "fair_value": round(fv, 2),
            "current_price": d["price"], "upside_pct": up, "signal": signal(up)}


def model_dcf(d):
    fcf = d["fcf"] or d["op_cf"]
    if not fcf or not d["shares"] or fcf <= 0:
        return {"available": False, "reason": "Fritt kassaflöde saknas eller negativt"}
    wacc = max(0.06, min(0.04 + (d["beta"] or 1.0) * 0.055, 0.18))
    g_high = min(0.12, max(0.03, d["growth_5y"] or 0.08))
    g_low  = g_high * 0.5
    g_term = 0.025
    fcf_ps = fcf / d["shares"]
    pv, cf = 0.0, fcf_ps
    for yr in range(1, 11):
        cf *= (1 + (g_high if yr <= 5 else g_low))
        pv += cf / ((1 + wacc) ** yr)
    terminal = (cf * (1 + g_term)) / (wacc - g_term)
    intrinsic = pv + terminal / ((1 + wacc) ** 10)
    up = upside_pct(intrinsic, d["price"])
    return {"available": True, "model": "DCF-värdering",
            "intrinsic_value": round(intrinsic, 2),
            "buy_below_20pct_mos": round(intrinsic * 0.80, 2),
            "buy_below_30pct_mos": round(intrinsic * 0.70, 2),
            "current_price": d["price"], "upside_pct": up, "signal": signal(up),
            "assumptions": {"wacc_pct": round(wacc * 100, 1),
                            "growth_yr1_5_pct": round(g_high * 100, 1),
                            "growth_yr6_10_pct": round(g_low * 100, 1),
                            "terminal_growth_pct": round(g_term * 100, 1),
                            "fcf_per_share": round(fcf_ps, 2)}}


def model_graham(d):
    if not d["eps"] or not d["book_value"] or d["eps"] <= 0 or d["book_value"] <= 0:
        return {"available": False, "reason": "EPS eller bokfört värde saknas/negativt"}
    intrinsic = math.sqrt(22.5 * d["eps"] * d["book_value"])
    up = upside_pct(intrinsic, d["price"])
    return {"available": True, "model": "Graham-formel",
            "intrinsic_value": round(intrinsic, 2),
            "current_price": d["price"], "upside_pct": up, "signal": signal(up),
            "inputs": {"eps": d["eps"], "book_value_per_share": round(d["book_value"], 2)}}


def model_relative(d, peers):
    if not peers:
        return {"available": False, "reason": "Inga peers"}
    def avg(lst):
        vals = [x for x in lst if x is not None]
        return round(sum(vals) / len(vals), 1) if vals else None
    avg_pe = avg([p.get("forward_pe") for p in peers])
    avg_pb = avg([p.get("pb") for p in peers])
    details = {}
    if avg_pe and d["eps"] and d["eps"] > 0:
        details["pe_based_fair_value"] = round(d["eps"] * avg_pe, 2)
        details["peer_avg_pe"] = avg_pe
    if avg_pb and d["book_value"] and d["book_value"] > 0:
        details["pb_based_fair_value"] = round(d["book_value"] * avg_pb, 2)
        details["peer_avg_pb"] = avg_pb
    fvs = [v for k, v in details.items() if "fair_value" in k]
    avg_fv = round(sum(fvs) / len(fvs), 2) if fvs else None
    up = upside_pct(avg_fv, d["price"])
    return {"available": True, "model": "Relativvärdering",
            "avg_fair_value": avg_fv, "current_price": d["price"],
            "upside_pct": up, "signal": signal(up),
            "peer_averages": {"pe": avg_pe, "pb": avg_pb},
            "details": details, "n_peers": len(peers)}


def build_summary(price, models):
    fvs, sigs = [], []
    for m in models:
        if not m.get("available"):
            continue
        fv = m.get("intrinsic_value") or m.get("fair_value") or m.get("avg_fair_value")
        if fv:
            fvs.append(fv)
        if m.get("signal"):
            sigs.append(m["signal"])
    if not fvs:
        return {"avg_fair_value": None, "consensus": "OTILLRÄCKLIG DATA"}
    avg_fv = round(sum(fvs) / len(fvs), 2)
    up = upside_pct(avg_fv, price)
    buy = sigs.count("KÖPVÄRD")
    sell = sigs.count("DYR")
    consensus = "KÖPVÄRD" if buy > sell else "DYR" if sell > buy else "NEUTRAL"
    return {"avg_fair_value": avg_fv, "upside_pct": up, "consensus": consensus,
            "model_signals": {"köpvärd": buy, "neutral": sigs.count("NEUTRAL"), "dyr": sell}}


# ── PDF-LOGIK ─────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    text_parts = []
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        tmp_path = f.name
    try:
        with pdfplumber.open(tmp_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    text_parts.append(f"[Sida {i+1}]\n{text}")
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        rows = [" | ".join([str(c or "") for c in row]) for row in table if row]
                        if rows:
                            text_parts.append(f"[Tabell sida {i+1}]\n" + "\n".join(rows))
    finally:
        os.unlink(tmp_path)
    return "\n\n".join(text_parts)


def extract_financials_with_claude(text: str, company_hint: str = "") -> dict:
    client = get_claude()
    prompt = f"""Du är en finansanalytiker. Nedan är text från en eller flera kvartalsrapporter.
Extrahera nyckeltal och returnera ENDAST ett JSON-objekt, inget annat, inga backticks.

Bolag: {company_hint or "Okänt"}

REGLER:
- Använd senaste tillgängliga 12-månaders (TTM) värden
- Om kvartalsdata finns, summera 4 kvartal för årstal
- Ange belopp i originalvaluta och enhet (miljoner/miljarder)
- Om ett värde saknas, sätt null
- Returnera BARA JSON

{{
  "company_name": "",
  "currency": "",
  "reporting_unit": "miljoner",
  "fiscal_year_end": null,
  "revenue": null,
  "gross_profit": null,
  "ebitda": null,
  "operating_income": null,
  "net_income": null,
  "eps": null,
  "free_cash_flow": null,
  "operating_cash_flow": null,
  "total_assets": null,
  "total_debt": null,
  "cash_and_equivalents": null,
  "total_equity": null,
  "shares_outstanding": null,
  "dividend_per_share": null,
  "revenue_growth_yoy": null,
  "sector": null,
  "beta": null,
  "notes": ""
}}

Rapporttext:
{text[:12000]}"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = message.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"Kunde inte tolka svar från AI: {raw[:200]}")


def build_financials_dict(data: dict, current_price: float | None) -> dict:
    unit = data.get("reporting_unit", "miljoner").lower()
    multiplier = 1_000_000
    if "miljard" in unit or "billion" in unit:
        multiplier = 1_000_000_000
    elif "tusen" in unit or "thousand" in unit:
        multiplier = 1_000

    def scale(val):
        if val is None:
            return None
        try:
            return float(val) * multiplier
        except (TypeError, ValueError):
            return None

    def sf(val):
        if val is None:
            return None
        try:
            return float(val)
        except:
            return None

    shares   = scale(data.get("shares_outstanding"))
    equity   = scale(data.get("total_equity"))
    bv_ps    = (equity / shares) if (equity and shares and shares > 0) else None
    debt     = scale(data.get("total_debt"))
    cash     = scale(data.get("cash_and_equivalents"))
    fcf      = scale(data.get("free_cash_flow"))
    op_cf    = scale(data.get("operating_cash_flow"))
    ebitda   = scale(data.get("ebitda"))
    revenue  = scale(data.get("revenue"))
    net_inc  = scale(data.get("net_income"))

    market_cap = (current_price * shares) if (current_price and shares) else None
    ev = (market_cap + (debt or 0) - (cash or 0)) if market_cap else None

    return dict(
        ticker="PDF", name=data.get("company_name", "Okänt bolag"),
        sector=data.get("sector") or "Okänd", industry="Okänd",
        currency=data.get("currency", "?"), country="",
        price=current_price, market_cap=market_cap, enterprise_value=ev,
        eps=sf(data.get("eps")), revenue=revenue, net_income=net_inc,
        ebitda=ebitda, gross_profit=scale(data.get("gross_profit")),
        book_value=bv_ps, total_debt=debt, cash=cash, shares=shares,
        fcf=fcf, op_cf=op_cf,
        beta=sf(data.get("beta")) or 1.0,
        growth_5y=sf(data.get("revenue_growth_yoy")),
        forward_pe=None, peg=None,
    )


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "OK", "message": "Värderingsmotor körs. Prova /value/AAPL, /compare?tickers=AAPL,MSFT eller POST /value-from-report"}


@app.get("/value/{ticker}")
def value_stock(ticker: str):
    try:
        d = fetch_data(ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Kunde inte hämta data för {ticker}: {e}")
    if not d["price"]:
        raise HTTPException(status_code=404, detail=f"Ingen kursinformation för {ticker}")
    pe_r  = model_pe(d)
    pb_r  = model_pb(d)
    ev_r  = model_ev_ebitda(d)
    dcf_r = model_dcf(d)
    gr_r  = model_graham(d)
    all_m = [pe_r, pb_r, ev_r, dcf_r, gr_r]
    return {
        "company": {"ticker": d["ticker"], "name": d["name"], "sector": d["sector"],
                    "industry": d["industry"], "country": d["country"],
                    "currency": d["currency"], "current_price": d["price"],
                    "market_cap": d["market_cap"]},
        "summary": build_summary(d["price"], all_m),
        "models": {"pe": pe_r, "pb": pb_r, "ev_ebitda": ev_r, "dcf": dcf_r, "graham": gr_r},
    }


@app.get("/compare")
def compare_stocks(tickers: str):
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="Ange minst ett ticker-symbol")
    peer_data, results = [], []
    for t in ticker_list:
        try:
            d = fetch_data(t)
            peer_data.append({"ticker": t, "forward_pe": d["forward_pe"],
                "pb": d["price"] / d["book_value"] if d["price"] and d["book_value"] and d["book_value"] > 0 else None})
        except Exception:
            pass
    for t in ticker_list:
        try:
            d = fetch_data(t)
            peers = [p for p in peer_data if p["ticker"] != t]
            rel_r = model_relative(d, peers)
            pe_r  = model_pe(d)
            pb_r  = model_pb(d)
            ev_r  = model_ev_ebitda(d)
            dcf_r = model_dcf(d)
            gr_r  = model_graham(d)
            results.append({"ticker": t, "name": d["name"], "sector": d["sector"],
                "current_price": d["price"], "currency": d["currency"],
                "summary": build_summary(d["price"], [pe_r, pb_r, ev_r, dcf_r, gr_r, rel_r]),
                "relative_valuation": rel_r,
                "models": {"pe": pe_r, "pb": pb_r, "ev_ebitda": ev_r, "dcf": dcf_r, "graham": gr_r}})
        except Exception as e:
            results.append({"ticker": t, "error": str(e)})
    return {"comparison": results, "n_stocks": len(results)}


@app.post("/value-from-report")
async def value_from_report(
    files: List[UploadFile] = File(...),
    current_price: Optional[float] = Form(None),
    company_name: Optional[str] = Form(""),
):
    """
    Ladda upp en eller flera kvartalsrapporter (PDF).
    Claude läser ut nyckeltalen och kör alla värderingsmodeller.
    
    - files: en eller flera PDF-filer
    - current_price: aktuell aktiekurs (valfri, krävs för P/E och P/B)
    - company_name: hjälper AI:n identifiera bolaget (valfri)
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY saknas i miljövariabler")

    # Extrahera text från alla PDFer
    all_text_parts = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} är inte en PDF")
        content = await file.read()
        if len(content) > 20 * 1024 * 1024:  # 20 MB max
            raise HTTPException(status_code=400, detail=f"{file.filename} är för stor (max 20 MB)")
        try:
            text = extract_text_from_pdf(content)
            all_text_parts.append(f"=== {file.filename} ===\n{text}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Kunde inte läsa {file.filename}: {e}")

    combined_text = "\n\n".join(all_text_parts)
    if not combined_text.strip():
        raise HTTPException(status_code=400, detail="Ingen läsbar text hittades i PDF:erna")

    # Låt Claude extrahera nyckeltal
    try:
        raw_financials = extract_financials_with_claude(combined_text, company_name or "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI-extraktion misslyckades: {e}")

    # Bygg finansobjekt och kör modeller
    try:
        d = build_financials_dict(raw_financials, current_price)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kunde inte normalisera finansdata: {e}")

    pe_r  = model_pe(d)
    pb_r  = model_pb(d)
    ev_r  = model_ev_ebitda(d)
    dcf_r = model_dcf(d)
    gr_r  = model_graham(d)
    all_m = [pe_r, pb_r, ev_r, dcf_r, gr_r]

    return {
        "source": "PDF-rapport",
        "files_processed": [f.filename for f in files],
        "company": {
            "name": d["name"],
            "currency": d["currency"],
            "sector": d["sector"],
            "current_price": d.get("price"),
            "reporting_unit": raw_financials.get("reporting_unit"),
        },
        "extracted_financials": {
            "revenue": d.get("revenue"),
            "ebitda": d.get("ebitda"),
            "net_income": d.get("net_income"),
            "eps": d.get("eps"),
            "free_cash_flow": d.get("fcf"),
            "book_value_per_share": d.get("book_value"),
            "total_debt": d.get("total_debt"),
            "cash": d.get("cash"),
            "shares_outstanding": d.get("shares"),
        },
        "ai_notes": raw_financials.get("notes"),
        "summary": build_summary(d.get("price"), all_m),
        "models": {"pe": pe_r, "pb": pb_r, "ev_ebitda": ev_r, "dcf": dcf_r, "graham": gr_r},
        "warning": "Värderingen baseras på AI-extraherad data. Verifiera mot originalrapporten." if not current_price else
                   "Notera: utan marknadskurs kan P/E och P/B inte beräknas fullt ut.",
    }
