from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import math

app = FastAPI(title="Värderingsmotor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def safe(val, default=None):
    """Return None if value is missing or non-finite."""
    try:
        if val is None:
            return default
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def fetch_data(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info

    # --- Income statement ---
    eps          = safe(info.get("trailingEps"))
    revenue      = safe(info.get("totalRevenue"))
    net_income   = safe(info.get("netIncomeToCommon"))
    ebitda       = safe(info.get("ebitda"))
    gross_profit = safe(info.get("grossProfits"))

    # --- Balance sheet ---
    book_value   = safe(info.get("bookValue"))          # per share
    total_debt   = safe(info.get("totalDebt"))
    cash         = safe(info.get("totalCash"))
    shares       = safe(info.get("sharesOutstanding"))

    # --- Cash flow ---
    fcf          = safe(info.get("freeCashflow"))
    op_cf        = safe(info.get("operatingCashflow"))

    # --- Market data ---
    price        = safe(info.get("currentPrice") or info.get("regularMarketPrice"))
    market_cap   = safe(info.get("marketCap"))
    beta         = safe(info.get("beta"), 1.0)

    # --- Growth & sector ---
    growth_5y    = safe(info.get("earningsGrowth"))     # trailing 12m
    forward_pe   = safe(info.get("forwardPE"))
    peg          = safe(info.get("pegRatio"))
    sector       = info.get("sector", "Okänd")
    industry     = info.get("industry", "Okänd")
    name         = info.get("longName") or info.get("shortName") or ticker
    currency     = info.get("currency", "USD")
    country      = info.get("country", "")

    enterprise_value = None
    if market_cap and total_debt and cash:
        enterprise_value = market_cap + total_debt - cash
    elif market_cap:
        enterprise_value = market_cap

    return dict(
        ticker=ticker.upper(), name=name, sector=sector, industry=industry,
        currency=currency, country=country, price=price, market_cap=market_cap,
        enterprise_value=enterprise_value, eps=eps, revenue=revenue,
        net_income=net_income, ebitda=ebitda, gross_profit=gross_profit,
        book_value=book_value, total_debt=total_debt, cash=cash, shares=shares,
        fcf=fcf, op_cf=op_cf, beta=beta, growth_5y=growth_5y,
        forward_pe=forward_pe, peg=peg,
    )


# ── VALUATION MODELS ─────────────────────────────────────────────────────────

def model_pe(d: dict) -> dict:
    """P/E-värdering: jämför aktuell kurs mot historiska PE-snitt."""
    if not d["eps"] or not d["price"] or d["eps"] <= 0:
        return {"available": False, "reason": "EPS saknas eller negativt"}

    pe = d["price"] / d["eps"]
    # Historiska snitt: S&P 500 ~16–18x, tillväxtbolag 25–35x
    sector_benchmarks = {
        "Technology": 28, "Healthcare": 22, "Financial Services": 14,
        "Consumer Cyclical": 20, "Consumer Defensive": 18,
        "Energy": 12, "Utilities": 16, "Industrials": 18,
        "Basic Materials": 14, "Real Estate": 30, "Communication Services": 22,
    }
    benchmark = sector_benchmarks.get(d["sector"], 18)
    fair_value = d["eps"] * benchmark
    upside = ((fair_value - d["price"]) / d["price"]) * 100

    signal = "KÖPVÄRD" if upside > 15 else "NEUTRAL" if upside > -10 else "DYR"

    return {
        "available": True,
        "model": "P/E-värdering",
        "current_pe": round(pe, 1),
        "sector_benchmark_pe": benchmark,
        "fair_value": round(fair_value, 2),
        "current_price": d["price"],
        "upside_pct": round(upside, 1),
        "signal": signal,
        "note": f"Benchmarkar mot sektorssnitt ({d['sector']}): {benchmark}x",
    }


def model_pb(d: dict) -> dict:
    """P/B-värdering: pris i förhållande till bokfört värde."""
    if not d["book_value"] or not d["price"] or d["book_value"] <= 0:
        return {"available": False, "reason": "Bokfört värde saknas"}

    pb = d["price"] / d["book_value"]
    sector_benchmarks = {
        "Technology": 6, "Healthcare": 4, "Financial Services": 1.3,
        "Consumer Cyclical": 3, "Consumer Defensive": 4,
        "Energy": 1.5, "Utilities": 1.5, "Industrials": 2.5,
        "Basic Materials": 1.8, "Real Estate": 1.4, "Communication Services": 3,
    }
    benchmark = sector_benchmarks.get(d["sector"], 2.5)
    fair_value = d["book_value"] * benchmark
    upside = ((fair_value - d["price"]) / d["price"]) * 100
    signal = "KÖPVÄRD" if upside > 15 else "NEUTRAL" if upside > -10 else "DYR"

    return {
        "available": True,
        "model": "P/B-värdering",
        "current_pb": round(pb, 2),
        "sector_benchmark_pb": benchmark,
        "book_value_per_share": round(d["book_value"], 2),
        "fair_value": round(fair_value, 2),
        "current_price": d["price"],
        "upside_pct": round(upside, 1),
        "signal": signal,
    }


def model_ev_ebitda(d: dict) -> dict:
    """EV/EBITDA-värdering."""
    if not d["ebitda"] or not d["enterprise_value"] or not d["shares"] or d["ebitda"] <= 0:
        return {"available": False, "reason": "EBITDA eller EV saknas"}

    ev_ebitda = d["enterprise_value"] / d["ebitda"]
    sector_benchmarks = {
        "Technology": 20, "Healthcare": 15, "Financial Services": 10,
        "Consumer Cyclical": 12, "Consumer Defensive": 13,
        "Energy": 7, "Utilities": 10, "Industrials": 12,
        "Basic Materials": 9, "Real Estate": 18, "Communication Services": 15,
    }
    benchmark = sector_benchmarks.get(d["sector"], 12)
    fair_ev = d["ebitda"] * benchmark
    equity_value = fair_ev - (d["total_debt"] or 0) + (d["cash"] or 0)
    fair_value_per_share = equity_value / d["shares"] if d["shares"] else None
    upside = ((fair_value_per_share - d["price"]) / d["price"]) * 100 if fair_value_per_share and d["price"] else None
    signal = "KÖPVÄRD" if (upside or 0) > 15 else "NEUTRAL" if (upside or 0) > -10 else "DYR"

    return {
        "available": True,
        "model": "EV/EBITDA-värdering",
        "current_ev_ebitda": round(ev_ebitda, 1),
        "sector_benchmark": benchmark,
        "fair_value": round(fair_value_per_share, 2) if fair_value_per_share else None,
        "current_price": d["price"],
        "upside_pct": round(upside, 1) if upside is not None else None,
        "signal": signal,
    }


def model_dcf(d: dict) -> dict:
    """DCF-värdering baserad på fritt kassaflöde."""
    fcf = d["fcf"] or d["op_cf"]
    if not fcf or not d["shares"] or fcf <= 0:
        return {"available": False, "reason": "Fritt kassaflöde saknas eller negativt"}

    # Antaganden
    risk_free = 0.04          # 10-årig statsobligation ~4%
    equity_premium = 0.055
    wacc = risk_free + (d["beta"] or 1.0) * equity_premium
    wacc = max(0.06, min(wacc, 0.18))   # klämmer mellan 6–18%

    g_high = min(0.12, max(0.03, (d["growth_5y"] or 0.08)))   # fas 1: år 1–5
    g_low  = g_high * 0.5                                       # fas 2: år 6–10
    g_terminal = 0.025                                          # terminal tillväxt

    fcf_ps = fcf / d["shares"]   # per aktie
    pv = 0.0

    cf = fcf_ps
    for yr in range(1, 11):
        g = g_high if yr <= 5 else g_low
        cf = cf * (1 + g)
        pv += cf / ((1 + wacc) ** yr)

    terminal = (cf * (1 + g_terminal)) / (wacc - g_terminal)
    pv_terminal = terminal / ((1 + wacc) ** 10)
    intrinsic = pv + pv_terminal

    # Margin of safety
    mos_20 = intrinsic * 0.80
    mos_30 = intrinsic * 0.70

    upside = ((intrinsic - d["price"]) / d["price"]) * 100 if d["price"] else None
    signal = "KÖPVÄRD" if (upside or 0) > 15 else "NEUTRAL" if (upside or 0) > -10 else "DYR"

    return {
        "available": True,
        "model": "DCF-värdering",
        "intrinsic_value": round(intrinsic, 2),
        "buy_below_20pct_mos": round(mos_20, 2),
        "buy_below_30pct_mos": round(mos_30, 2),
        "current_price": d["price"],
        "upside_pct": round(upside, 1) if upside is not None else None,
        "signal": signal,
        "assumptions": {
            "wacc_pct": round(wacc * 100, 1),
            "growth_yr1_5_pct": round(g_high * 100, 1),
            "growth_yr6_10_pct": round(g_low * 100, 1),
            "terminal_growth_pct": round(g_terminal * 100, 1),
            "fcf_per_share": round(fcf_ps, 2),
        },
    }


def model_graham(d: dict) -> dict:
    """Grahams intrinsic value: sqrt(22.5 × EPS × BVPS)."""
    if not d["eps"] or not d["book_value"] or d["eps"] <= 0 or d["book_value"] <= 0:
        return {"available": False, "reason": "EPS eller bokfört värde saknas/negativt"}

    intrinsic = math.sqrt(22.5 * d["eps"] * d["book_value"])
    upside = ((intrinsic - d["price"]) / d["price"]) * 100 if d["price"] else None
    signal = "KÖPVÄRD" if (upside or 0) > 15 else "NEUTRAL" if (upside or 0) > -10 else "DYR"

    return {
        "available": True,
        "model": "Graham-formel",
        "intrinsic_value": round(intrinsic, 2),
        "current_price": d["price"],
        "upside_pct": round(upside, 1) if upside is not None else None,
        "signal": signal,
        "inputs": {"eps": d["eps"], "book_value_per_share": d["book_value"]},
        "note": "Passar bäst för stabila, lönsamma bolag utan hög skuldsättning",
    }


def model_relative(d: dict, peers: list[dict]) -> dict:
    """Relativvärdering: jämför bolagets multiplar mot en grupp peers."""
    if not peers:
        return {"available": False, "reason": "Inga peers att jämföra med"}

    def avg(lst):
        vals = [x for x in lst if x is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    peer_pes    = [p["forward_pe"] for p in peers]
    peer_ebitdas = [p.get("ev_ebitda") for p in peers]
    peer_pbs    = [p["pb"] for p in peers]

    avg_pe   = avg(peer_pes)
    avg_ev   = avg(peer_ebitdas)
    avg_pb   = avg(peer_pbs)

    # Beräkna fair values baserat på peer-snitt
    results = {}

    if avg_pe and d["eps"] and d["eps"] > 0:
        fv = d["eps"] * avg_pe
        results["pe_based_fair_value"] = round(fv, 2)
        results["peer_avg_pe"] = avg_pe

    if avg_pb and d["book_value"] and d["book_value"] > 0:
        fv = d["book_value"] * avg_pb
        results["pb_based_fair_value"] = round(fv, 2)
        results["peer_avg_pb"] = avg_pb

    fair_values = [v for k, v in results.items() if "fair_value" in k]
    avg_fair = round(sum(fair_values) / len(fair_values), 2) if fair_values else None
    upside = ((avg_fair - d["price"]) / d["price"]) * 100 if avg_fair and d["price"] else None
    signal = "KÖPVÄRD" if (upside or 0) > 15 else "NEUTRAL" if (upside or 0) > -10 else "DYR"

    return {
        "available": True,
        "model": "Relativvärdering",
        "avg_fair_value": avg_fair,
        "current_price": d["price"],
        "upside_pct": round(upside, 1) if upside is not None else None,
        "signal": signal,
        "peer_averages": {"pe": avg_pe, "ev_ebitda": avg_ev, "pb": avg_pb},
        "details": results,
        "n_peers": len(peers),
    }


def build_summary(price, models: list[dict]) -> dict:
    """Viktat snitt av tillgängliga fair values."""
    fair_values = []
    signals = []
    for m in models:
        if not m.get("available"):
            continue
        fv = m.get("intrinsic_value") or m.get("fair_value") or m.get("avg_fair_value")
        if fv:
            fair_values.append(fv)
        if m.get("signal"):
            signals.append(m["signal"])

    if not fair_values:
        return {"avg_fair_value": None, "consensus": "OTILLRÄCKLIG DATA"}

    avg_fv = round(sum(fair_values) / len(fair_values), 2)
    upside = round(((avg_fv - price) / price) * 100, 1) if price else None

    buy_count  = signals.count("KÖPVÄRD")
    sell_count = signals.count("DYR")
    consensus  = "KÖPVÄRD" if buy_count > sell_count else "DYR" if sell_count > buy_count else "NEUTRAL"

    return {
        "avg_fair_value": avg_fv,
        "upside_pct": upside,
        "consensus": consensus,
        "model_signals": {"köpvärd": buy_count, "neutral": signals.count("NEUTRAL"), "dyr": sell_count},
    }


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "OK", "message": "Värderingsmotor körs. Prova /value/AAPL eller /compare?tickers=AAPL,MSFT"}


@app.get("/value/{ticker}")
def value_stock(ticker: str):
    try:
        d = fetch_data(ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Kunde inte hämta data för {ticker}: {e}")

    if not d["price"]:
        raise HTTPException(status_code=404, detail=f"Ingen kursinformation hittades för {ticker}")

    pe_res     = model_pe(d)
    pb_res     = model_pb(d)
    ev_res     = model_ev_ebitda(d)
    dcf_res    = model_dcf(d)
    graham_res = model_graham(d)

    all_models = [pe_res, pb_res, ev_res, dcf_res, graham_res]
    summary    = build_summary(d["price"], all_models)

    return {
        "company": {
            "ticker": d["ticker"],
            "name": d["name"],
            "sector": d["sector"],
            "industry": d["industry"],
            "country": d["country"],
            "currency": d["currency"],
            "current_price": d["price"],
            "market_cap": d["market_cap"],
        },
        "summary": summary,
        "models": {
            "pe":        pe_res,
            "pb":        pb_res,
            "ev_ebitda": ev_res,
            "dcf":       dcf_res,
            "graham":    graham_res,
        },
    }


@app.get("/compare")
def compare_stocks(tickers: str):
    """Jämför flera bolag: /compare?tickers=AAPL,MSFT,GOOGL"""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="Ange minst ett ticker-symbol")

    results = []
    peer_data = []

    for t in ticker_list:
        try:
            d = fetch_data(t)
            peer_data.append({
                "ticker": t,
                "forward_pe": d["forward_pe"],
                "pb": d["price"] / d["book_value"] if d["price"] and d["book_value"] and d["book_value"] > 0 else None,
                "ev_ebitda": d["enterprise_value"] / d["ebitda"] if d["enterprise_value"] and d["ebitda"] and d["ebitda"] > 0 else None,
            })
        except Exception:
            pass

    for t in ticker_list:
        try:
            d = fetch_data(t)
            peers = [p for p in peer_data if p["ticker"] != t]
            relative = model_relative(d, peers)

            pe_res     = model_pe(d)
            pb_res     = model_pb(d)
            ev_res     = model_ev_ebitda(d)
            dcf_res    = model_dcf(d)
            graham_res = model_graham(d)
            all_models = [pe_res, pb_res, ev_res, dcf_res, graham_res, relative]
            summary    = build_summary(d["price"], all_models)

            results.append({
                "ticker": t,
                "name": d["name"],
                "sector": d["sector"],
                "current_price": d["price"],
                "currency": d["currency"],
                "summary": summary,
                "relative_valuation": relative,
                "models": {
                    "pe": pe_res, "pb": pb_res,
                    "ev_ebitda": ev_res, "dcf": dcf_res, "graham": graham_res,
                },
            })
        except Exception as e:
            results.append({"ticker": t, "error": str(e)})

    return {"comparison": results, "n_stocks": len(results)}
