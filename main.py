from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import math, os, json, time, requests

app = FastAPI(title="Värderingsmotor API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FMP_BASE  = "https://financialmodelingprep.com/stable"
TD_BASE   = "https://api.twelvedata.com"
_cache    = {}
CACHE_TTL = 300

def get_fmp_key():
    return os.environ.get("FMP_API_KEY", "abtaanU8oyPCYOSTOvM5oEaEVRjkmMTG")

def get_td_key():
    return os.environ.get("TWELVE_DATA_KEY", "fa13988488f6419288401068a898cc92")

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

def fmp_get(path, params={}):
    r = requests.get(f"{FMP_BASE}{path}", params={"apikey": get_fmp_key(), **params}, timeout=15)
    r.raise_for_status()
    return r.json()

def td_get(path, params={}):
    r = requests.get(f"{TD_BASE}{path}", params={"apikey": get_td_key(), **params}, timeout=15)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("status") == "error":
        raise ValueError(data.get("message", "Twelve Data fel"))
    return data

SECTOR_LOOKUP = {
    "AAPL":"Technology","MSFT":"Technology","GOOGL":"Technology","GOOG":"Technology",
    "META":"Technology","NVDA":"Technology","AMD":"Technology","INTC":"Technology",
    "ORCL":"Technology","CRM":"Technology","ADBE":"Technology","QCOM":"Technology",
    "AMZN":"Consumer Cyclical","TSLA":"Consumer Cyclical","NKE":"Consumer Cyclical",
    "MCD":"Consumer Defensive","KO":"Consumer Defensive","PG":"Consumer Defensive",
    "WMT":"Consumer Defensive","COST":"Consumer Defensive",
    "JPM":"Financial Services","BAC":"Financial Services","GS":"Financial Services",
    "MS":"Financial Services","V":"Financial Services","MA":"Financial Services",
    "JNJ":"Healthcare","PFE":"Healthcare","UNH":"Healthcare","ABBV":"Healthcare",
    "XOM":"Energy","CVX":"Energy","COP":"Energy",
    "CAT":"Industrials","BA":"Industrials","GE":"Industrials","HON":"Industrials",
    "VOLV-B.ST":"Industrials","SAND.ST":"Industrials","SKF-B.ST":"Industrials",
    "ATCO-A.ST":"Industrials","INVE-B.ST":"Financial Services",
    "ERIC-B.ST":"Communication Services","TEL2-B.ST":"Communication Services",
    "SEB-A.ST":"Financial Services","SHB-A.ST":"Financial Services",
    "ESSITY-B.ST":"Consumer Defensive","HM-B.ST":"Consumer Cyclical",
    "SPOT":"Communication Services","NFLX":"Communication Services",
    "DIS":"Communication Services","CMCSA":"Communication Services",
    "T":"Communication Services","VZ":"Communication Services",
}

def fetch_data(ticker: str) -> dict:
    ticker = ticker.upper()
    if ticker in _cache:
        ts, data = _cache[ticker]
        if time.time() - ts < CACHE_TTL:
            return data

    # --- Realtidskurs från Twelve Data ---
    price, name, currency = None, ticker, "USD"
    try:
        quote = td_get("/quote", {"symbol": ticker})
        price    = safe(quote.get("close"))
        name     = quote.get("name") or ticker
        currency = quote.get("currency") or "USD"
    except: pass

    # --- Fundamentaldata från FMP ---
    profile    = {}
    income     = {}
    balance    = {}
    cashflow   = {}
    ratios     = {}

    try:
        p = fmp_get("/profile", {"symbol": ticker})
        profile = p[0] if isinstance(p, list) and p else {}
        if not price:
            price = safe(profile.get("price"))
        if not name or name == ticker:
            name = profile.get("companyName") or ticker
        currency = profile.get("currency") or currency
    except: pass

    try:
        inc = fmp_get("/income-statement", {"symbol": ticker, "limit": 1})
        income = inc[0] if isinstance(inc, list) and inc else {}
    except: pass

    try:
        bal = fmp_get("/balance-sheet-statement", {"symbol": ticker, "limit": 1})
        balance = bal[0] if isinstance(bal, list) and bal else {}
    except: pass

    try:
        cf = fmp_get("/cash-flow-statement", {"symbol": ticker, "limit": 1})
        cashflow = cf[0] if isinstance(cf, list) and cf else {}
    except: pass

    try:
        rat = fmp_get("/ratios-ttm", {"symbol": ticker})
        ratios = rat[0] if isinstance(rat, list) and rat else {}
    except: pass

    # --- Mappning ---
    sector   = SECTOR_LOOKUP.get(ticker) or profile.get("sector") or "Okänd"
    industry = profile.get("industry") or "Okänd"
    country  = profile.get("country") or ("SE" if ticker.endswith(".ST") else "US")

    market_cap   = safe(profile.get("mktCap"))
    shares       = safe(profile.get("sharesOutstanding")) or safe(balance.get("commonStock"))
    beta         = safe(profile.get("beta")) or 1.0

    eps          = safe(ratios.get("netIncomePerShareTTM")) or safe(income.get("eps"))
    revenue      = safe(income.get("revenue"))
    net_income   = safe(income.get("netIncome"))
    ebitda       = safe(income.get("ebitda"))
    gross_profit = safe(income.get("grossProfit"))

    total_debt   = safe(balance.get("totalDebt"))
    cash         = safe(balance.get("cashAndCashEquivalents"))
    book_value_total = safe(balance.get("totalStockholdersEquity"))
    book_value   = (book_value_total / shares) if book_value_total and shares and shares > 0 else None

    fcf          = safe(cashflow.get("freeCashFlow"))
    op_cf        = safe(cashflow.get("operatingCashFlow"))

    net_margin   = safe(ratios.get("netProfitMarginTTM"))
    gross_margin = safe(ratios.get("grossProfitMarginTTM"))
    roe          = safe(ratios.get("returnOnEquityTTM"))
    growth       = safe(ratios.get("revenueGrowthTTM")) or safe(income.get("revenueGrowth")) or 0.05
    forward_pe   = safe(ratios.get("priceEarningsRatioTTM"))
    peg_ratio    = safe(ratios.get("priceEarningsToGrowthRatioTTM"))

    ev = None
    if market_cap and total_debt and cash:
        ev = market_cap + total_debt - cash
    elif market_cap:
        ev = market_cap

    result = dict(
        ticker=ticker, name=name, sector=sector, industry=industry,
        currency=currency, country=country, price=price, market_cap=market_cap,
        enterprise_value=ev, eps=eps, revenue=revenue, net_income=net_income,
        ebitda=ebitda, gross_profit=gross_profit, book_value=book_value,
        total_debt=total_debt, cash=cash, shares=shares, fcf=fcf, op_cf=op_cf,
        beta=beta, growth_5y=growth, forward_pe=forward_pe, peg_ratio=peg_ratio,
        roe=roe, net_margin=net_margin, gross_margin=gross_margin,
    )
    _cache[ticker] = (time.time(), result)
    return result

# ── SEKTORSSNITT ──────────────────────────────────────────────────────────────
SECTOR_PE = {"Technology":28,"Healthcare":22,"Financial Services":14,"Consumer Cyclical":20,"Consumer Defensive":18,"Energy":12,"Utilities":16,"Industrials":18,"Basic Materials":14,"Real Estate":30,"Communication Services":22}
SECTOR_PB = {"Technology":6,"Healthcare":4,"Financial Services":1.3,"Consumer Cyclical":3,"Consumer Defensive":4,"Energy":1.5,"Utilities":1.5,"Industrials":2.5,"Basic Materials":1.8,"Real Estate":1.4,"Communication Services":3}
SECTOR_EV = {"Technology":20,"Healthcare":15,"Financial Services":10,"Consumer Cyclical":12,"Consumer Defensive":13,"Energy":7,"Utilities":10,"Industrials":12,"Basic Materials":9,"Real Estate":18,"Communication Services":15}

# ── MODELLER ──────────────────────────────────────────────────────────────────
def model_dcf_eps(d):
    if not d["eps"] or not d["price"] or d["eps"] <= 0:
        return {"available":False,"reason":"EPS saknas eller negativt"}
    wacc   = max(0.07, min(0.04 + (d["beta"] or 1.0) * 0.055, 0.15))
    g_high = min(0.15, max(0.04, (d["growth_5y"] or 0.07) * 1.2))
    g_low  = g_high * 0.5
    g_term = 0.025
    eps = d["eps"]; pv, e = 0.0, eps
    for yr in range(1, 11):
        e *= (1 + (g_high if yr <= 5 else g_low))
        pv += e / ((1 + wacc) ** yr)
    terminal  = (e * (1 + g_term)) / (wacc - g_term)
    intrinsic = pv + terminal / ((1 + wacc) ** 10)
    up = upside_pct(intrinsic, d["price"])
    return {"available":True,"model":"DCF (vinstbaserad)","intrinsic_value":round(intrinsic,2),
            "buy_below_20pct_mos":round(intrinsic*0.80,2),"buy_below_30pct_mos":round(intrinsic*0.70,2),
            "current_price":d["price"],"upside_pct":up,"signal":signal(up),"weight":0.35,
            "assumptions":{"wacc_pct":round(wacc*100,1),"growth_yr1_5_pct":round(g_high*100,1),
                           "growth_yr6_10_pct":round(g_low*100,1),"terminal_growth_pct":round(g_term*100,1),"eps":round(eps,2)},
            "explanation":"Diskonterar framtida vinster till nuvärde – tre-fas modell"}

def model_dcf_fcf(d):
    fcf = d["fcf"] or d["op_cf"]
    if not fcf or not d["shares"] or fcf <= 0:
        return {"available":False,"reason":"Fritt kassaflöde saknas eller negativt"}
    wacc   = max(0.07, min(0.04 + (d["beta"] or 1.0) * 0.055, 0.15))
    g_high = min(0.12, max(0.03, d["growth_5y"] or 0.07))
    g_low  = g_high * 0.5; g_term = 0.025
    fcf_ps = fcf / d["shares"]; pv, cf = 0.0, fcf_ps
    for yr in range(1, 11):
        cf *= (1 + (g_high if yr <= 5 else g_low))
        pv += cf / ((1 + wacc) ** yr)
    intrinsic = pv + (cf*(1+g_term)/(wacc-g_term)) / ((1+wacc)**10)
    up = upside_pct(intrinsic, d["price"])
    return {"available":True,"model":"DCF (kassaflödesbaserad)","intrinsic_value":round(intrinsic,2),
            "buy_below_20pct_mos":round(intrinsic*0.80,2),"buy_below_30pct_mos":round(intrinsic*0.70,2),
            "current_price":d["price"],"upside_pct":up,"signal":signal(up),"weight":0.15,
            "assumptions":{"wacc_pct":round(wacc*100,1),"growth_yr1_5_pct":round(g_high*100,1),
                           "growth_yr6_10_pct":round(g_low*100,1),"terminal_growth_pct":round(g_term*100,1),
                           "fcf_per_share":round(fcf_ps,2)},
            "explanation":"Diskonterar framtida fria kassaflöden – mer konservativ metod"}

def model_pe(d):
    if not d["eps"] or not d["price"] or d["eps"] <= 0:
        return {"available":False,"reason":"EPS saknas eller negativt"}
    bm = SECTOR_PE.get(d["sector"], 18)
    roe = d.get("roe") or 0
    if roe > 0.30: bm = round(bm * 1.25, 1)
    elif roe > 0.20: bm = round(bm * 1.10, 1)
    fv = d["eps"] * bm; up = upside_pct(fv, d["price"])
    return {"available":True,"model":"P/E-värdering","current_pe":round(d["price"]/d["eps"],1),
            "sector_benchmark_pe":bm,"roe_adjusted":roe>0.20,"fair_value":round(fv,2),
            "current_price":d["price"],"upside_pct":up,"signal":signal(up),"weight":0.20,
            "explanation":"Aktiekurs / Vinst per aktie mot sektorssnitt (justerat för ROE)"}

def model_ev_ebitda(d):
    if not d["ebitda"] or not d["enterprise_value"] or not d["shares"] or d["ebitda"] <= 0:
        return {"available":False,"reason":"EBITDA eller EV saknas"}
    bm = SECTOR_EV.get(d["sector"], 12)
    growth = d.get("growth_5y") or 0
    if growth > 0.15: bm = round(bm * 1.20, 1)
    elif growth > 0.08: bm = round(bm * 1.10, 1)
    if d.get("revenue") and d["revenue"] > 0:
        if d["ebitda"] / d["revenue"] > 0.30: bm = round(bm * 1.10, 1)
    fair_ev = d["ebitda"] * bm
    eq = fair_ev - (d["total_debt"] or 0) + (d["cash"] or 0)
    fv = eq / d["shares"]; up = upside_pct(fv, d["price"])
    return {"available":True,"model":"EV/EBITDA","current_ev_ebitda":round(d["enterprise_value"]/d["ebitda"],1),
            "adjusted_benchmark":bm,"fair_value":round(fv,2),"current_price":d["price"],
            "upside_pct":up,"signal":signal(up),"weight":0.20,
            "explanation":"Företagsvärde / EBITDA – justerat för tillväxt och marginal"}

def model_pb(d):
    if not d["book_value"] or not d["price"] or d["book_value"] <= 0:
        return {"available":False,"reason":"Bokfört värde saknas","in_consensus":False}
    bm = SECTOR_PB.get(d["sector"], 2.5)
    current_pb = d["price"] / d["book_value"]
    in_consensus = current_pb < 10
    fv = d["book_value"] * bm; up = upside_pct(fv, d["price"])
    return {"available":True,"model":"P/B-värdering","current_pb":round(current_pb,1),
            "sector_benchmark_pb":bm,"book_value_per_share":round(d["book_value"],2),
            "fair_value":round(fv,2),"current_price":d["price"],"upside_pct":up,"signal":signal(up),
            "in_consensus":in_consensus,"weight":0.05 if in_consensus else 0,
            "note":"Exkluderad från konsensus – P/B missvisande för bolag med stora aktieåterköp" if not in_consensus else None,
            "explanation":"Aktiekurs / Bokfört värde per aktie"}

def model_graham(d):
    if not d["eps"] or not d["book_value"] or d["eps"] <= 0 or d["book_value"] <= 0:
        return {"available":False,"reason":"EPS eller bokfört värde saknas","in_consensus":False}
    intrinsic = math.sqrt(22.5 * d["eps"] * d["book_value"])
    up = upside_pct(intrinsic, d["price"])
    traditional = d["sector"] in ["Industrials","Consumer Defensive","Utilities","Energy","Basic Materials"]
    return {"available":True,"model":"Graham-formel","intrinsic_value":round(intrinsic,2),
            "current_price":d["price"],"upside_pct":up,"signal":signal(up),
            "in_consensus":traditional,"weight":0.10 if traditional else 0,
            "note":"Referensmodell – ej inkluderad i konsensus för tech/tillväxtbolag" if not traditional else None,
            "inputs":{"eps":d["eps"],"book_value_per_share":round(d["book_value"],2)},
            "explanation":"Benjamin Grahams klassiska formel: √(22.5 × EPS × Bokfört värde)"}

def model_peer_relative(subject, peers_data):
    if not peers_data: return {"available":False,"reason":"Inga peer-data tillgängliga","in_consensus":False}
    pe_m, ev_m = [], []
    for p in peers_data:
        if p.get("eps") and p.get("price") and p["eps"]>0: pe_m.append(p["price"]/p["eps"])
        if p.get("ebitda") and p.get("enterprise_value") and p["ebitda"]>0: ev_m.append(p["enterprise_value"]/p["ebitda"])
    def median(lst):
        if not lst: return None
        s=sorted(lst); n=len(s)
        return round(s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2,1)
    med_pe=median(pe_m); med_ev=median(ev_m)
    fvs, details = [], {}
    if med_pe and subject.get("eps") and subject["eps"]>0:
        fv=subject["eps"]*med_pe; fvs.append(fv)
        details["pe"]={"peer_median":med_pe,"fair_value":round(fv,2)}
    if med_ev and subject.get("ebitda") and subject["ebitda"]>0 and subject.get("shares"):
        fv=(subject["ebitda"]*med_ev-(subject.get("total_debt") or 0)+(subject.get("cash") or 0))/subject["shares"]
        fvs.append(fv); details["ev_ebitda"]={"peer_median":med_ev,"fair_value":round(fv,2)}
    if not fvs: return {"available":False,"reason":"Otillräcklig peer-data","in_consensus":False}
    avg_fv=round(sum(fvs)/len(fvs),2); up=upside_pct(avg_fv,subject.get("price"))
    return {"available":True,"model":"Relativvärdering (AI-peers)","avg_fair_value":avg_fv,
            "current_price":subject.get("price"),"upside_pct":up,"signal":signal(up),
            "in_consensus":True,"weight":0.10,"multiples_used":details,"peer_count":len(peers_data),
            "explanation":f"Median-multiplar från {len(peers_data)} AI-identifierade jämförbara bolag"}

def build_summary(price, models):
    weighted_fvs=[]; sigs=[]; total_weight=0
    for m in models:
        if not m.get("available"): continue
        if "in_consensus" in m and not m["in_consensus"]:
            if m.get("signal"): sigs.append(m["signal"])
            continue
        fv = m.get("intrinsic_value") or m.get("fair_value") or m.get("avg_fair_value")
        weight = m.get("weight", 0.10)
        if fv and weight > 0:
            weighted_fvs.append((fv, weight)); total_weight += weight
        if m.get("signal"): sigs.append(m["signal"])
    if not weighted_fvs: return {"avg_fair_value":None,"consensus":"OTILLRÄCKLIG DATA"}
    avg_fv = round(sum(fv*(w/total_weight) for fv,w in weighted_fvs), 2)
    up = upside_pct(avg_fv, price)
    buy=sigs.count("KÖPVÄRD"); sell=sigs.count("DYR")
    return {"avg_fair_value":avg_fv,"upside_pct":up,
            "consensus":"KÖPVÄRD" if buy>sell else "DYR" if sell>buy else "NEUTRAL",
            "model_signals":{"köpvärd":buy,"neutral":sigs.count("NEUTRAL"),"dyr":sell},
            "methodology":"Viktat snitt: DCF-EPS 35%, P/E 20%, EV/EBITDA 20%, DCF-FCF 15%, Relativvärdering 10%"}

def identify_peers(company: dict):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key: return [], "AI-peer-identifiering ej aktiverad"
    mc_str = f"${company['market_cap']/1e9:.0f}B" if company.get("market_cap") else "okänt"
    prompt = f"""Du är en senior aktieanalytiker. Identifiera 4 börsnoterade peer-bolag för:
Bolag: {company['name']} ({company['ticker']}), Sektor: {company['sector']}, Industri: {company['industry']}, Land: {company['country']}, Börsvärde: {mc_str}
Välj peers med liknande affärsmodell och storlek. Använd korrekta börstickers.
Returnera ENBART JSON: {{"peers": ["T1","T2","T3","T4"], "reasoning": "En mening"}}"""
    try:
        resp = requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key":api_key,"anthropic-version":"2023-06-01","content-type":"application/json"},
            json={"model":"claude-sonnet-4-20250514","max_tokens":200,"messages":[{"role":"user","content":prompt}]},
            timeout=20)
        raw = resp.json()["content"][0]["text"].strip().replace("```json","").replace("```","").strip()
        p = json.loads(raw)
        return p.get("peers",[]), p.get("reasoning","")
    except Exception as e:
        return [], f"Misslyckades: {e}"

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status":"OK","message":"Prova /value/AAPL, /value/GOOGL, /value/VOLV-B.ST"}

@app.get("/debug")
def debug():
    return {"fmp_key":bool(get_fmp_key()),"td_key":bool(get_td_key())}

@app.get("/value/{ticker}")
def value_stock(ticker: str):
    try: d = fetch_data(ticker.upper())
    except Exception as e: raise HTTPException(400, f"Kunde inte hämta data för {ticker}: {e}")
    if not d["price"]: raise HTTPException(404, f"Ingen kursinformation för {ticker}")
    dcf_eps=model_dcf_eps(d); dcf_fcf=model_dcf_fcf(d)
    pe_r=model_pe(d); ev_r=model_ev_ebitda(d); pb_r=model_pb(d); gr_r=model_graham(d)
    all_m=[dcf_eps,dcf_fcf,pe_r,ev_r,pb_r,gr_r]
    return {
        "company":{"ticker":d["ticker"],"name":d["name"],"sector":d["sector"],"industry":d["industry"],
                   "country":d["country"],"currency":d["currency"],"current_price":d["price"],"market_cap":d["market_cap"]},
        "key_metrics":{"roe":d.get("roe"),"net_margin":d.get("net_margin"),"gross_margin":d.get("gross_margin"),
                       "revenue_growth":d.get("growth_5y"),"beta":d.get("beta"),
                       "forward_pe":d.get("forward_pe"),"peg_ratio":d.get("peg_ratio")},
        "summary":build_summary(d["price"],all_m),
        "models":{"dcf_eps":dcf_eps,"dcf_fcf":dcf_fcf,"pe":pe_r,"ev_ebitda":ev_r,"pb":pb_r,"graham":gr_r}
    }

@app.get("/value-with-peers/{ticker}")
def value_with_peers(ticker: str):
    try: d = fetch_data(ticker.upper())
    except Exception as e: raise HTTPException(400, f"Kunde inte hämta data för {ticker}: {e}")
    if not d["price"]: raise HTTPException(404, f"Ingen kursinformation för {ticker}")
    peer_tickers, peer_reasoning = identify_peers(d)
    peers_data, peer_errors = [], []
    for pt in peer_tickers:
        try:
            pd = fetch_data(pt)
            if pd.get("price"): peers_data.append(pd)
        except Exception as e: peer_errors.append(f"{pt}: {e}")
    dcf_eps=model_dcf_eps(d); dcf_fcf=model_dcf_fcf(d)
    pe_r=model_pe(d); ev_r=model_ev_ebitda(d); pb_r=model_pb(d); gr_r=model_graham(d)
    rel_r=model_peer_relative(d,peers_data)
    all_m=[dcf_eps,dcf_fcf,pe_r,ev_r,pb_r,gr_r,rel_r]
    peer_summary=[{
        "ticker":p["ticker"],"name":p["name"],"price":p["price"],"currency":p["currency"],
        "pe":round(p["price"]/p["eps"],1) if p.get("eps") and p["eps"]>0 else None,
        "pb":round(p["price"]/p["book_value"],1) if p.get("book_value") and p["book_value"]>0 else None,
        "ev_ebitda":round(p["enterprise_value"]/p["ebitda"],1) if p.get("ebitda") and p["ebitda"]>0 and p.get("enterprise_value") else None,
    } for p in peers_data]
    return {
        "company":{"ticker":d["ticker"],"name":d["name"],"sector":d["sector"],"industry":d["industry"],
                   "country":d["country"],"currency":d["currency"],"current_price":d["price"],"market_cap":d["market_cap"]},
        "key_metrics":{"roe":d.get("roe"),"net_margin":d.get("net_margin"),"gross_margin":d.get("gross_margin"),
                       "revenue_growth":d.get("growth_5y"),"beta":d.get("beta"),
                       "forward_pe":d.get("forward_pe"),"peg_ratio":d.get("peg_ratio")},
        "summary":build_summary(d["price"],all_m),
        "models":{"dcf_eps":dcf_eps,"dcf_fcf":dcf_fcf,"pe":pe_r,"ev_ebitda":ev_r,"pb":pb_r,"graham":gr_r,"relative":rel_r},
        "peer_analysis":{"identified_peers":peer_tickers,"reasoning":peer_reasoning,
                         "peers_data":peer_summary,"fetch_errors":peer_errors}
    }

@app.get("/compare")
def compare_stocks(tickers: str):
    ticker_list=[t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list: raise HTTPException(400,"Ange minst ett ticker-symbol")
    results=[]
    for t in ticker_list:
        try:
            d=fetch_data(t)
            dcf_eps=model_dcf_eps(d);dcf_fcf=model_dcf_fcf(d)
            pe_r=model_pe(d);ev_r=model_ev_ebitda(d);pb_r=model_pb(d);gr_r=model_graham(d)
            results.append({"ticker":t,"name":d["name"],"sector":d["sector"],"current_price":d["price"],
                "currency":d["currency"],"summary":build_summary(d["price"],[dcf_eps,dcf_fcf,pe_r,ev_r,pb_r,gr_r]),
                "models":{"dcf_eps":dcf_eps,"dcf_fcf":dcf_fcf,"pe":pe_r,"ev_ebitda":ev_r,"pb":pb_r,"graham":gr_r}})
        except Exception as e: results.append({"ticker":t,"error":str(e)})
    return {"comparison":results,"n_stocks":len(results)}
