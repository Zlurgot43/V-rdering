from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import requests, math, os, json, anthropic

app = FastAPI(title="Värderingsmotor API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FMP_KEY   = os.environ.get("FMP_API_KEY", "demo")
FMP_BASE  = "https://financialmodelingprep.com/stable"
_claude   = None

def get_claude():
    global _claude
    if _claude is None:
        _claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _claude

def fmp(path, extra={}):
    r = requests.get(f"{FMP_BASE}{path}", params={"apikey": FMP_KEY, **extra}, timeout=15)
    r.raise_for_status()
    return r.json()

def safe(val, default=None):
    try:
        if val is None: return default
        f = float(val)
        return f if math.isfinite(f) else default
    except: return default

def signal(upside):
    if upside is None: return "OKÄND"
    return "KÖPVÄRD" if upside > 15 else "NEUTRAL" if upside > -10 else "DYR"

def upside_pct(fair, price):
    if fair and price: return round(((fair - price) / price) * 100, 1)
    return None

def fetch_data(ticker):
    ticker = ticker.upper()
    q_data = fmp("/quote", {"symbol": ticker})
    if not q_data or isinstance(q_data, dict): raise ValueError(f"Ingen data för {ticker}")
    q = q_data[0] if isinstance(q_data, list) else q_data

    def safe_get(path, key):
        try:
            d = fmp(path, {"symbol": ticker, "limit": 1})
            return d[0] if isinstance(d, list) and d else {}
        except: return {}

    profile  = safe_get("/profile", "sector") or {}
    cf       = safe_get("/cash-flow-statement", "freeCashFlow")
    bs       = safe_get("/balance-sheet-statement", "totalDebt")
    inc      = safe_get("/income-statement", "ebitda")
    try:
        ratios_d = fmp("/ratios-ttm", {"symbol": ticker})
        ratios = ratios_d[0] if isinstance(ratios_d, list) and ratios_d else {}
    except: ratios = {}

    price      = safe(q.get("price"))
    market_cap = safe(q.get("marketCap"))
    shares     = safe(q.get("sharesOutstanding"))
    eps        = safe(q.get("eps"))
    bv_total   = safe(bs.get("totalStockholdersEquity"))
    book_value = (bv_total / shares) if (bv_total and shares and shares > 0) else safe(ratios.get("bookValuePerShareTTM"))
    total_debt = safe(bs.get("totalDebt"))
    cash       = safe(bs.get("cashAndCashEquivalents"))
    ebitda     = safe(inc.get("ebitda")) or safe(q.get("ebitda"))
    revenue    = safe(inc.get("revenue"))
    net_income = safe(inc.get("netIncome"))
    fcf        = safe(cf.get("freeCashFlow"))
    op_cf      = safe(cf.get("operatingCashFlow"))
    ev         = (market_cap + (total_debt or 0) - (cash or 0)) if market_cap and total_debt and cash else market_cap

    return dict(
        ticker=ticker,
        name=profile.get("companyName") or q.get("name") or ticker,
        sector=profile.get("sector") or "Okänd",
        industry=profile.get("industry") or "Okänd",
        currency=profile.get("currency") or "USD",
        country=profile.get("country") or "",
        price=price, market_cap=market_cap, enterprise_value=ev,
        eps=eps, revenue=revenue, net_income=net_income, ebitda=ebitda,
        book_value=book_value, total_debt=total_debt, cash=cash, shares=shares,
        fcf=fcf, op_cf=op_cf,
        beta=safe(profile.get("beta"), 1.0),
        growth_5y=safe(ratios.get("revenueGrowthTTM"), 0.08),
        forward_pe=safe(ratios.get("priceEarningsRatioTTM")), peg=None,
    )

SECTOR_PE = {"Technology":28,"Healthcare":22,"Financial Services":14,"Consumer Cyclical":20,"Consumer Defensive":18,"Energy":12,"Utilities":16,"Industrials":18,"Basic Materials":14,"Real Estate":30,"Communication Services":22}
SECTOR_PB = {"Technology":6,"Healthcare":4,"Financial Services":1.3,"Consumer Cyclical":3,"Consumer Defensive":4,"Energy":1.5,"Utilities":1.5,"Industrials":2.5,"Basic Materials":1.8,"Real Estate":1.4,"Communication Services":3}
SECTOR_EV = {"Technology":20,"Healthcare":15,"Financial Services":10,"Consumer Cyclical":12,"Consumer Defensive":13,"Energy":7,"Utilities":10,"Industrials":12,"Basic Materials":9,"Real Estate":18,"Communication Services":15}

def model_pe(d):
    if not d["eps"] or not d["price"] or d["eps"]<=0: return {"available":False,"reason":"EPS saknas eller negativt"}
    bm=SECTOR_PE.get(d["sector"],18); fv=d["eps"]*bm; up=upside_pct(fv,d["price"])
    return {"available":True,"model":"P/E-värdering","current_pe":round(d["price"]/d["eps"],1),"sector_benchmark_pe":bm,"fair_value":round(fv,2),"current_price":d["price"],"upside_pct":up,"signal":signal(up),"note":f"Sektorssnitt ({d['sector']}): {bm}x"}

def model_pb(d):
    if not d["book_value"] or not d["price"] or d["book_value"]<=0: return {"available":False,"reason":"Bokfört värde saknas"}
    bm=SECTOR_PB.get(d["sector"],2.5); fv=d["book_value"]*bm; up=upside_pct(fv,d["price"])
    return {"available":True,"model":"P/B-värdering","current_pb":round(d["price"]/d["book_value"],2),"sector_benchmark_pb":bm,"book_value_per_share":round(d["book_value"],2),"fair_value":round(fv,2),"current_price":d["price"],"upside_pct":up,"signal":signal(up)}

def model_ev_ebitda(d):
    if not d["ebitda"] or not d["enterprise_value"] or not d["shares"] or d["ebitda"]<=0: return {"available":False,"reason":"EBITDA eller EV saknas"}
    bm=SECTOR_EV.get(d["sector"],12); fair_ev=d["ebitda"]*bm
    eq=fair_ev-(d["total_debt"] or 0)+(d["cash"] or 0); fv=eq/d["shares"]; up=upside_pct(fv,d["price"])
    return {"available":True,"model":"EV/EBITDA-värdering","current_ev_ebitda":round(d["enterprise_value"]/d["ebitda"],1),"sector_benchmark":bm,"fair_value":round(fv,2),"current_price":d["price"],"upside_pct":up,"signal":signal(up)}

def model_dcf(d):
    fcf=d["fcf"] or d["op_cf"]
    if not fcf or not d["shares"] or fcf<=0: return {"available":False,"reason":"Fritt kassaflöde saknas eller negativt"}
    wacc=max(0.06,min(0.04+(d["beta"] or 1.0)*0.055,0.18))
    g_high=min(0.12,max(0.03,d["growth_5y"] or 0.08)); g_low=g_high*0.5; g_term=0.025
    fcf_ps=fcf/d["shares"]; pv,cf=0.0,fcf_ps
    for yr in range(1,11):
        cf*=(1+(g_high if yr<=5 else g_low)); pv+=cf/((1+wacc)**yr)
    intrinsic=pv+(cf*(1+g_term)/(wacc-g_term))/((1+wacc)**10)
    up=upside_pct(intrinsic,d["price"])
    return {"available":True,"model":"DCF-värdering","intrinsic_value":round(intrinsic,2),"buy_below_20pct_mos":round(intrinsic*0.80,2),"buy_below_30pct_mos":round(intrinsic*0.70,2),"current_price":d["price"],"upside_pct":up,"signal":signal(up),"assumptions":{"wacc_pct":round(wacc*100,1),"growth_yr1_5_pct":round(g_high*100,1),"terminal_growth_pct":round(g_term*100,1),"fcf_per_share":round(fcf_ps,2)}}

def model_graham(d):
    if not d["eps"] or not d["book_value"] or d["eps"]<=0 or d["book_value"]<=0: return {"available":False,"reason":"EPS eller bokfört värde saknas/negativt"}
    intrinsic=math.sqrt(22.5*d["eps"]*d["book_value"]); up=upside_pct(intrinsic,d["price"])
    return {"available":True,"model":"Graham-formel","intrinsic_value":round(intrinsic,2),"current_price":d["price"],"upside_pct":up,"signal":signal(up),"inputs":{"eps":d["eps"],"book_value_per_share":round(d["book_value"],2)}}

def model_peer_relative(subject, peers_data):
    """
    Relativvärdering mot AI-identifierade peers.
    Beräknar median-multiplar för P/E, P/B, EV/EBITDA bland peers
    och värderar subject-bolaget mot dessa.
    """
    if not peers_data: return {"available":False,"reason":"Inga peer-data tillgängliga"}

    pe_multiples, pb_multiples, ev_multiples = [], [], []

    for p in peers_data:
        if p.get("eps") and p.get("price") and p["eps"] > 0:
            pe_multiples.append(p["price"] / p["eps"])
        if p.get("book_value") and p.get("price") and p["book_value"] > 0:
            pb_multiples.append(p["price"] / p["book_value"])
        if p.get("ebitda") and p.get("enterprise_value") and p["ebitda"] > 0:
            ev_multiples.append(p["enterprise_value"] / p["ebitda"])

    def median(lst):
        if not lst: return None
        s = sorted(lst)
        n = len(s)
        return round(s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2, 1)

    med_pe = median(pe_multiples)
    med_pb = median(pb_multiples)
    med_ev = median(ev_multiples)

    fair_values = []
    details = {}

    if med_pe and subject.get("eps") and subject["eps"] > 0:
        fv = subject["eps"] * med_pe
        fair_values.append(fv)
        details["pe"] = {"peer_median": med_pe, "fair_value": round(fv, 2), "n_peers": len(pe_multiples)}

    if med_pb and subject.get("book_value") and subject["book_value"] > 0:
        fv = subject["book_value"] * med_pb
        fair_values.append(fv)
        details["pb"] = {"peer_median": med_pb, "fair_value": round(fv, 2), "n_peers": len(pb_multiples)}

    if med_ev and subject.get("ebitda") and subject["ebitda"] > 0 and subject.get("shares"):
        fair_ev = subject["ebitda"] * med_ev
        eq = fair_ev - (subject.get("total_debt") or 0) + (subject.get("cash") or 0)
        fv = eq / subject["shares"]
        fair_values.append(fv)
        details["ev_ebitda"] = {"peer_median": med_ev, "fair_value": round(fv, 2), "n_peers": len(ev_multiples)}

    if not fair_values: return {"available":False,"reason":"Otillräcklig data för peer-jämförelse"}

    avg_fv = round(sum(fair_values) / len(fair_values), 2)
    up = upside_pct(avg_fv, subject.get("price"))

    return {
        "available": True,
        "model": "Relativvärdering (AI-peers)",
        "avg_fair_value": avg_fv,
        "current_price": subject.get("price"),
        "upside_pct": up,
        "signal": signal(up),
        "multiples_used": details,
        "peer_count": len(peers_data),
        "note": f"Baserat på median-multiplar från {len(peers_data)} AI-identifierade peers"
    }

def build_summary(price, models):
    fvs, sigs = [], []
    for m in models:
        if not m.get("available"): continue
        fv = m.get("intrinsic_value") or m.get("fair_value") or m.get("avg_fair_value")
        if fv: fvs.append(fv)
        if m.get("signal"): sigs.append(m["signal"])
    if not fvs: return {"avg_fair_value":None,"consensus":"OTILLRÄCKLIG DATA"}
    avg_fv = round(sum(fvs)/len(fvs), 2)
    up = upside_pct(avg_fv, price)
    buy = sigs.count("KÖPVÄRD"); sell = sigs.count("DYR")
    return {"avg_fair_value":avg_fv,"upside_pct":up,"consensus":"KÖPVÄRD" if buy>sell else "DYR" if sell>buy else "NEUTRAL","model_signals":{"köpvärd":buy,"neutral":sigs.count("NEUTRAL"),"dyr":sell}}


def identify_peers_with_claude(company: dict) -> tuple[list[str], str]:
    """
    Låt Claude identifiera relevanta peer-bolag baserat på bolagsprofilen.
    Returnerar (lista med tickers, motivering).
    """
    client = get_claude()

    mc_str = f"${company['market_cap']/1e9:.0f}B" if company.get("market_cap") else "okänt"

    prompt = f"""Du är en senior aktieanalytiker på en investmentbank.
Identifiera 4-5 börsnoterade peer-bolag för relativvärdering av:

Bolag: {company['name']}
Ticker: {company['ticker']}
Sektor: {company['sector']}
Industri: {company['industry']}
Land: {company['country']}
Börsvärde: {mc_str}

Välj peers som:
- Har liknande affärsmodell och konkurrerar på samma marknad
- Är börsnoterade och har tillräcklig likviditet
- Är jämförbara i storlek (±80% av börsvärde)
- Täcker geografisk spridning om relevant

Returnera ENBART JSON (inga backticks):
{{
  "peers": ["TICKER1", "TICKER2", "TICKER3", "TICKER4"],
  "reasoning": "En mening om varför dessa valdes och vad de har gemensamt"
}}

Viktigt: Använd korrekta börstickers (t.ex. VOLV-B.ST för Volvo B på Stockholmsbörsen, ERIC-B.ST för Ericsson B)."""

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role":"user","content":prompt}]
    )
    raw = msg.content[0].text.strip().replace("```json","").replace("```","").strip()
    parsed = json.loads(raw)
    return parsed.get("peers", []), parsed.get("reasoning", "")


@app.get("/")
def root():
    return {"status":"OK","message":"Prova /value/AAPL, /value-with-peers/AAPL eller /compare?tickers=AAPL,MSFT"}

@app.get("/value/{ticker}")
def value_stock(ticker: str):
    try: d = fetch_data(ticker.upper())
    except Exception as e: raise HTTPException(400, f"Kunde inte hämta data för {ticker}: {e}")
    if not d["price"]: raise HTTPException(404, f"Ingen kursinformation för {ticker}")
    pe_r=model_pe(d); pb_r=model_pb(d); ev_r=model_ev_ebitda(d); dcf_r=model_dcf(d); gr_r=model_graham(d)
    all_m=[pe_r,pb_r,ev_r,dcf_r,gr_r]
    return {"company":{"ticker":d["ticker"],"name":d["name"],"sector":d["sector"],"industry":d["industry"],"country":d["country"],"currency":d["currency"],"current_price":d["price"],"market_cap":d["market_cap"]},"summary":build_summary(d["price"],all_m),"models":{"pe":pe_r,"pb":pb_r,"ev_ebitda":ev_r,"dcf":dcf_r,"graham":gr_r}}

@app.get("/value-with-peers/{ticker}")
def value_with_peers(ticker: str):
    """
    Komplett värdering med AI-identifierade peers och automatisk relativvärdering.
    Claude identifierar relevanta peer-bolag, hämtar deras data och kör relativvärdering.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(500, "ANTHROPIC_API_KEY saknas – relativvärdering med AI-peers kräver denna nyckel")

    # Steg 1: Hämta data för huvudbolaget
    try: d = fetch_data(ticker.upper())
    except Exception as e: raise HTTPException(400, f"Kunde inte hämta data för {ticker}: {e}")
    if not d["price"]: raise HTTPException(404, f"Ingen kursinformation för {ticker}")

    # Steg 2: Låt Claude identifiera peers
    peer_tickers, peer_reasoning = [], "Inga peers identifierade"
    peer_fetch_errors = []
    try:
        peer_tickers, peer_reasoning = identify_peers_with_claude(d)
    except Exception as e:
        peer_fetch_errors.append(f"Peer-identifiering misslyckades: {e}")

    # Steg 3: Hämta data för alla peers
    peers_data = []
    for pt in peer_tickers:
        try:
            pd = fetch_data(pt)
            if pd.get("price"):
                peers_data.append(pd)
        except Exception as e:
            peer_fetch_errors.append(f"{pt}: {e}")

    # Steg 4: Kör alla värderingsmodeller
    pe_r   = model_pe(d)
    pb_r   = model_pb(d)
    ev_r   = model_ev_ebitda(d)
    dcf_r  = model_dcf(d)
    gr_r   = model_graham(d)
    rel_r  = model_peer_relative(d, peers_data)
    all_m  = [pe_r, pb_r, ev_r, dcf_r, gr_r, rel_r]

    # Peer-sammanfattning för transparens
    peer_summary = []
    for pd in peers_data:
        pe_val = round(pd["price"]/pd["eps"],1) if pd.get("eps") and pd["eps"]>0 else None
        pb_val = round(pd["price"]/pd["book_value"],1) if pd.get("book_value") and pd["book_value"]>0 else None
        ev_val = round(pd["enterprise_value"]/pd["ebitda"],1) if pd.get("ebitda") and pd["ebitda"]>0 and pd.get("enterprise_value") else None
        peer_summary.append({
            "ticker": pd["ticker"], "name": pd["name"],
            "price": pd["price"], "currency": pd["currency"],
            "pe": pe_val, "pb": pb_val, "ev_ebitda": ev_val,
        })

    return {
        "company": {"ticker":d["ticker"],"name":d["name"],"sector":d["sector"],"industry":d["industry"],"country":d["country"],"currency":d["currency"],"current_price":d["price"],"market_cap":d["market_cap"]},
        "summary": build_summary(d["price"], all_m),
        "models": {"pe":pe_r,"pb":pb_r,"ev_ebitda":ev_r,"dcf":dcf_r,"graham":gr_r,"relative":rel_r},
        "peer_analysis": {
            "identified_peers": peer_tickers,
            "reasoning": peer_reasoning,
            "peers_data": peer_summary,
            "fetch_errors": peer_fetch_errors,
        }
    }

@app.get("/compare")
def compare_stocks(tickers: str):
    ticker_list=[t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list: raise HTTPException(400,"Ange minst ett ticker-symbol")
    peer_data,results=[],[]
    for t in ticker_list:
        try:
            d=fetch_data(t)
            peer_data.append({"ticker":t,"forward_pe":d["forward_pe"],"pb":d["price"]/d["book_value"] if d["price"] and d.get("book_value") and d["book_value"]>0 else None})
        except: pass
    for t in ticker_list:
        try:
            d=fetch_data(t)
            peers=[p for p in peer_data if p["ticker"]!=t]
            pe_r=model_pe(d);pb_r=model_pb(d);ev_r=model_ev_ebitda(d);dcf_r=model_dcf(d);gr_r=model_graham(d)
            results.append({"ticker":t,"name":d["name"],"sector":d["sector"],"current_price":d["price"],"currency":d["currency"],"summary":build_summary(d["price"],[pe_r,pb_r,ev_r,dcf_r,gr_r]),"models":{"pe":pe_r,"pb":pb_r,"ev_ebitda":ev_r,"dcf":dcf_r,"graham":gr_r}})
        except Exception as e: results.append({"ticker":t,"error":str(e)})
    return {"comparison":results,"n_stocks":len(results)}
