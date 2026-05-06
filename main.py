from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import math, os, json, time, requests

app = FastAPI(title="Värderingsmotor API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Cache för att undvika för många anrop
_cache = {}
CACHE_TTL = 300  # 5 minuter

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

# Headers som efterliknar en webbläsare – undviker Yahoo-blockering
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Origin": "https://finance.yahoo.com",
    "Referer": "https://finance.yahoo.com/",
}

def yahoo_fetch(ticker: str) -> dict:
    """Hämtar data direkt från Yahoo Finance JSON-API med browser-headers."""
    ticker = ticker.upper()

    # Kolla cache
    if ticker in _cache:
        ts, data = _cache[ticker]
        if time.time() - ts < CACHE_TTL:
            return data

    session = requests.Session()
    session.headers.update(HEADERS)

    # Steg 1: Hämta crumb (krävs av Yahoo sedan 2023)
    crumb_url = "https://query2.finance.yahoo.com/v1/test/getcrumb"
    try:
        cr = session.get(crumb_url, timeout=10)
        crumb = cr.text.strip()
    except:
        crumb = ""

    # Steg 2: Hämta quoteSummary med alla moduler
    modules = "price,summaryDetail,financialData,defaultKeyStatistics,incomeStatementHistory,cashflowStatementHistory,balanceSheetHistory"
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
    params = {"modules": modules, "crumb": crumb, "formatted": "false"}

    resp = session.get(url, params=params, timeout=15)
    if resp.status_code != 200:
        raise ValueError(f"Yahoo svarade med {resp.status_code} för {ticker}")

    result = resp.json().get("quoteSummary", {}).get("result", [])
    if not result:
        raise ValueError(f"Ingen data för {ticker}")

    d = result[0]
    price_data = d.get("price", {})
    fin_data   = d.get("financialData", {})
    key_stats  = d.get("defaultKeyStatistics", {})
    summary    = d.get("summaryDetail", {})

    # Hämta från income statement (TTM)
    inc_stmts = d.get("incomeStatementHistory", {}).get("incomeStatementHistory", [])
    inc = inc_stmts[0] if inc_stmts else {}

    # Hämta från cashflow (TTM)
    cf_stmts = d.get("cashflowStatementHistory", {}).get("cashflowStatements", [])
    cf = cf_stmts[0] if cf_stmts else {}

    # Hämta från balance sheet
    bs_stmts = d.get("balanceSheetHistory", {}).get("balanceSheetStatements", [])
    bs = bs_stmts[0] if bs_stmts else {}

    def g(obj, key):
        """Hämtar råvärde från Yahoo-objekt (hanterar både {raw: x} och direktvärde)."""
        v = obj.get(key)
        if isinstance(v, dict): return safe(v.get("raw"))
        return safe(v)

    price      = g(price_data, "regularMarketPrice")
    market_cap = g(price_data, "marketCap")
    shares     = g(key_stats, "sharesOutstanding")
    eps        = g(key_stats, "trailingEps")
    book_value = g(key_stats, "bookValue")
    beta       = g(key_stats, "beta") or 1.0
    forward_pe = g(key_stats, "forwardPE")

    revenue    = g(inc, "totalRevenue")
    net_income = g(inc, "netIncome")
    ebitda     = g(fin_data, "ebitda")
    gross_profit = g(inc, "grossProfit")

    fcf        = g(fin_data, "freeCashflow")
    op_cf      = g(cf, "totalCashFromOperatingActivities")

    total_debt = g(fin_data, "totalDebt")
    cash       = g(fin_data, "totalCash")

    roe        = g(fin_data, "returnOnEquity")
    roa        = g(fin_data, "returnOnAssets")
    net_margin = g(fin_data, "profitMargins")
    gross_margin = g(fin_data, "grossMargins")
    growth     = g(fin_data, "revenueGrowth") or 0.05

    ev = None
    if market_cap and total_debt and cash:
        ev = market_cap + total_debt - cash
    elif market_cap:
        ev = market_cap

    sector   = price_data.get("sector") or d.get("summaryProfile", {}).get("sector") or "Okänd"
    industry = price_data.get("industry") or d.get("summaryProfile", {}).get("industry") or "Okänd"
    name     = price_data.get("longName") or price_data.get("shortName") or ticker
    currency = price_data.get("currency") or "USD"
    country  = d.get("summaryProfile", {}).get("country") or ""

    result_dict = dict(
        ticker=ticker, name=name, sector=sector, industry=industry,
        currency=currency, country=country, price=price, market_cap=market_cap,
        enterprise_value=ev, eps=eps, revenue=revenue, net_income=net_income,
        ebitda=ebitda, gross_profit=gross_profit, book_value=book_value,
        total_debt=total_debt, cash=cash, shares=shares, fcf=fcf, op_cf=op_cf,
        beta=beta, growth_5y=growth, forward_pe=forward_pe,
        roe=roe, roa=roa, net_margin=net_margin, gross_margin=gross_margin,
    )

    _cache[ticker] = (time.time(), result_dict)
    return result_dict

# ── SEKTORSSNITT ──────────────────────────────────────────────────────────────
SECTOR_PE = {"Technology":28,"Healthcare":22,"Financial Services":14,"Consumer Cyclical":20,"Consumer Defensive":18,"Energy":12,"Utilities":16,"Industrials":18,"Basic Materials":14,"Real Estate":30,"Communication Services":22}
SECTOR_PB = {"Technology":6,"Healthcare":4,"Financial Services":1.3,"Consumer Cyclical":3,"Consumer Defensive":4,"Energy":1.5,"Utilities":1.5,"Industrials":2.5,"Basic Materials":1.8,"Real Estate":1.4,"Communication Services":3}
SECTOR_EV = {"Technology":20,"Healthcare":15,"Financial Services":10,"Consumer Cyclical":12,"Consumer Defensive":13,"Energy":7,"Utilities":10,"Industrials":12,"Basic Materials":9,"Real Estate":18,"Communication Services":15}

# ── MODELLER ──────────────────────────────────────────────────────────────────
def model_pe(d):
    if not d["eps"] or not d["price"] or d["eps"] <= 0:
        return {"available":False,"reason":"EPS saknas eller negativt"}
    bm = SECTOR_PE.get(d["sector"], 18)
    fv = d["eps"] * bm; up = upside_pct(fv, d["price"])
    return {"available":True,"model":"P/E-värdering","current_pe":round(d["price"]/d["eps"],1),
            "sector_benchmark_pe":bm,"fair_value":round(fv,2),"current_price":d["price"],
            "upside_pct":up,"signal":signal(up),
            "explanation":"Värderar bolaget baserat på vinst per aktie mot sektorns P/E-snitt"}

def model_pb(d):
    if not d["book_value"] or not d["price"] or d["book_value"] <= 0:
        return {"available":False,"reason":"Bokfört värde saknas"}
    bm = SECTOR_PB.get(d["sector"], 2.5)
    fv = d["book_value"] * bm; up = upside_pct(fv, d["price"])
    return {"available":True,"model":"P/B-värdering","current_pb":round(d["price"]/d["book_value"],2),
            "sector_benchmark_pb":bm,"book_value_per_share":round(d["book_value"],2),
            "fair_value":round(fv,2),"current_price":d["price"],"upside_pct":up,"signal":signal(up),
            "explanation":"Jämför aktiekursen mot bolagets bokförda tillgångar per aktie"}

def model_ev_ebitda(d):
    if not d["ebitda"] or not d["enterprise_value"] or not d["shares"] or d["ebitda"] <= 0:
        return {"available":False,"reason":"EBITDA eller EV saknas"}
    bm = SECTOR_EV.get(d["sector"], 12)
    fair_ev = d["ebitda"] * bm
    eq = fair_ev - (d["total_debt"] or 0) + (d["cash"] or 0)
    fv = eq / d["shares"]; up = upside_pct(fv, d["price"])
    return {"available":True,"model":"EV/EBITDA-värdering",
            "current_ev_ebitda":round(d["enterprise_value"]/d["ebitda"],1),
            "sector_benchmark":bm,"fair_value":round(fv,2),"current_price":d["price"],
            "upside_pct":up,"signal":signal(up),
            "explanation":"Värderar hela företaget inklusive skulder mot rörelseresultat"}

def model_dcf(d):
    fcf = d["fcf"] or d["op_cf"]
    if not fcf or not d["shares"] or fcf <= 0:
        return {"available":False,"reason":"Fritt kassaflöde saknas eller negativt"}
    wacc   = max(0.07, min(0.04 + (d["beta"] or 1.0) * 0.055, 0.16))
    g_high = min(0.12, max(0.03, d["growth_5y"] or 0.07))
    g_low  = g_high * 0.5; g_term = 0.025
    fcf_ps = fcf / d["shares"]; pv, cf = 0.0, fcf_ps
    for yr in range(1, 11):
        cf *= (1 + (g_high if yr <= 5 else g_low))
        pv += cf / ((1 + wacc) ** yr)
    intrinsic = pv + (cf*(1+g_term)/(wacc-g_term)) / ((1+wacc)**10)
    up = upside_pct(intrinsic, d["price"])
    return {"available":True,"model":"DCF-analys","intrinsic_value":round(intrinsic,2),
            "buy_below_20pct_mos":round(intrinsic*0.80,2),"buy_below_30pct_mos":round(intrinsic*0.70,2),
            "current_price":d["price"],"upside_pct":up,"signal":signal(up),
            "assumptions":{"wacc_pct":round(wacc*100,1),"growth_yr1_5_pct":round(g_high*100,1),
                           "growth_yr6_10_pct":round(g_low*100,1),"terminal_growth_pct":round(g_term*100,1),
                           "fcf_per_share":round(fcf_ps,2)},
            "explanation":"Diskonterar framtida kassaflöden till nuvärde – den mest fundamentala värderingsmetoden"}

def model_graham(d):
    if not d["eps"] or not d["book_value"] or d["eps"] <= 0 or d["book_value"] <= 0:
        return {"available":False,"reason":"EPS eller bokfört värde saknas/negativt"}
    intrinsic = math.sqrt(22.5 * d["eps"] * d["book_value"])
    up = upside_pct(intrinsic, d["price"])
    return {"available":True,"model":"Graham-formel","intrinsic_value":round(intrinsic,2),
            "current_price":d["price"],"upside_pct":up,"signal":signal(up),
            "inputs":{"eps":d["eps"],"book_value_per_share":round(d["book_value"],2)},
            "explanation":"Benjamin Grahams klassiska formel: √(22.5 × EPS × Bokfört värde)"}

def model_peer_relative(subject, peers_data):
    if not peers_data: return {"available":False,"reason":"Inga peer-data tillgängliga"}
    pe_m, pb_m, ev_m = [], [], []
    for p in peers_data:
        if p.get("eps") and p.get("price") and p["eps"] > 0: pe_m.append(p["price"]/p["eps"])
        if p.get("book_value") and p.get("price") and p["book_value"] > 0: pb_m.append(p["price"]/p["book_value"])
        if p.get("ebitda") and p.get("enterprise_value") and p["ebitda"] > 0: ev_m.append(p["enterprise_value"]/p["ebitda"])
    def median(lst):
        if not lst: return None
        s=sorted(lst); n=len(s)
        return round(s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2, 1)
    med_pe=median(pe_m); med_pb=median(pb_m); med_ev=median(ev_m)
    fvs, details = [], {}
    if med_pe and subject.get("eps") and subject["eps"]>0:
        fv=subject["eps"]*med_pe; fvs.append(fv)
        details["pe"]={"peer_median":med_pe,"fair_value":round(fv,2)}
    if med_pb and subject.get("book_value") and subject["book_value"]>0:
        fv=subject["book_value"]*med_pb; fvs.append(fv)
        details["pb"]={"peer_median":med_pb,"fair_value":round(fv,2)}
    if med_ev and subject.get("ebitda") and subject["ebitda"]>0 and subject.get("shares"):
        fv=(subject["ebitda"]*med_ev-(subject.get("total_debt") or 0)+(subject.get("cash") or 0))/subject["shares"]
        fvs.append(fv); details["ev_ebitda"]={"peer_median":med_ev,"fair_value":round(fv,2)}
    if not fvs: return {"available":False,"reason":"Otillräcklig data för peer-jämförelse"}
    avg_fv=round(sum(fvs)/len(fvs),2); up=upside_pct(avg_fv,subject.get("price"))
    return {"available":True,"model":"Relativvärdering (AI-peers)","avg_fair_value":avg_fv,
            "current_price":subject.get("price"),"upside_pct":up,"signal":signal(up),
            "multiples_used":details,"peer_count":len(peers_data),
            "explanation":f"Värderar mot median-multiplar från {len(peers_data)} AI-identifierade peers"}

def build_summary(price, models):
    fvs, sigs = [], []
    for m in models:
        if not m.get("available"): continue
        fv = m.get("intrinsic_value") or m.get("fair_value") or m.get("avg_fair_value")
        if fv: fvs.append(fv)
        if m.get("signal"): sigs.append(m["signal"])
    if not fvs: return {"avg_fair_value":None,"consensus":"OTILLRÄCKLIG DATA"}
    avg_fv=round(sum(fvs)/len(fvs),2); up=upside_pct(avg_fv,price)
    buy=sigs.count("KÖPVÄRD"); sell=sigs.count("DYR")
    return {"avg_fair_value":avg_fv,"upside_pct":up,
            "consensus":"KÖPVÄRD" if buy>sell else "DYR" if sell>buy else "NEUTRAL",
            "model_signals":{"köpvärd":buy,"neutral":sigs.count("NEUTRAL"),"dyr":sell}}

def identify_peers(company: dict):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key: return [], "AI-peer-identifiering ej aktiverad"
    mc_str = f"${company['market_cap']/1e9:.0f}B" if company.get("market_cap") else "okänt"
    prompt = f"""Du är en senior aktieanalytiker. Identifiera 4 börsnoterade peer-bolag för:
Bolag: {company['name']} ({company['ticker']}), Sektor: {company['sector']}, Industri: {company['industry']}, Land: {company['country']}, Börsvärde: {mc_str}
Välj peers med liknande affärsmodell och storlek. Använd korrekta börstickers (t.ex. VOLV-B.ST för svenska bolag).
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
    return {"status":"OK","message":"Prova /value/AAPL eller /value-with-peers/AAPL"}

@app.get("/value/{ticker}")
def value_stock(ticker: str):
    try: d = yahoo_fetch(ticker.upper())
    except Exception as e: raise HTTPException(400, f"Kunde inte hämta data för {ticker}: {e}")
    if not d["price"]: raise HTTPException(404, f"Ingen kursinformation för {ticker}")
    pe_r=model_pe(d); pb_r=model_pb(d); ev_r=model_ev_ebitda(d)
    dcf_r=model_dcf(d); gr_r=model_graham(d); all_m=[pe_r,pb_r,ev_r,dcf_r,gr_r]
    return {
        "company":{"ticker":d["ticker"],"name":d["name"],"sector":d["sector"],"industry":d["industry"],"country":d["country"],"currency":d["currency"],"current_price":d["price"],"market_cap":d["market_cap"]},
        "key_metrics":{"roe":d.get("roe"),"net_margin":d.get("net_margin"),"gross_margin":d.get("gross_margin"),"revenue_growth":d.get("growth_5y"),"beta":d.get("beta")},
        "summary":build_summary(d["price"],all_m),
        "models":{"pe":pe_r,"pb":pb_r,"ev_ebitda":ev_r,"dcf":dcf_r,"graham":gr_r}
    }

@app.get("/value-with-peers/{ticker}")
def value_with_peers(ticker: str):
    try: d = yahoo_fetch(ticker.upper())
    except Exception as e: raise HTTPException(400, f"Kunde inte hämta data för {ticker}: {e}")
    if not d["price"]: raise HTTPException(404, f"Ingen kursinformation för {ticker}")

    peer_tickers, peer_reasoning = identify_peers(d)
    peers_data, peer_errors = [], []
    for pt in peer_tickers:
        try:
            pd = yahoo_fetch(pt)
            if pd.get("price"): peers_data.append(pd)
        except Exception as e: peer_errors.append(f"{pt}: {e}")

    pe_r=model_pe(d); pb_r=model_pb(d); ev_r=model_ev_ebitda(d)
    dcf_r=model_dcf(d); gr_r=model_graham(d); rel_r=model_peer_relative(d,peers_data)
    all_m=[pe_r,pb_r,ev_r,dcf_r,gr_r,rel_r]

    peer_summary=[{
        "ticker":p["ticker"],"name":p["name"],"price":p["price"],"currency":p["currency"],
        "pe":round(p["price"]/p["eps"],1) if p.get("eps") and p["eps"]>0 else None,
        "pb":round(p["price"]/p["book_value"],1) if p.get("book_value") and p["book_value"]>0 else None,
        "ev_ebitda":round(p["enterprise_value"]/p["ebitda"],1) if p.get("ebitda") and p["ebitda"]>0 and p.get("enterprise_value") else None,
    } for p in peers_data]

    return {
        "company":{"ticker":d["ticker"],"name":d["name"],"sector":d["sector"],"industry":d["industry"],"country":d["country"],"currency":d["currency"],"current_price":d["price"],"market_cap":d["market_cap"]},
        "key_metrics":{"roe":d.get("roe"),"net_margin":d.get("net_margin"),"gross_margin":d.get("gross_margin"),"revenue_growth":d.get("growth_5y"),"beta":d.get("beta")},
        "summary":build_summary(d["price"],all_m),
        "models":{"pe":pe_r,"pb":pb_r,"ev_ebitda":ev_r,"dcf":dcf_r,"graham":gr_r,"relative":rel_r},
        "peer_analysis":{"identified_peers":peer_tickers,"reasoning":peer_reasoning,"peers_data":peer_summary,"fetch_errors":peer_errors}
    }

@app.get("/compare")
def compare_stocks(tickers: str):
    ticker_list=[t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list: raise HTTPException(400,"Ange minst ett ticker-symbol")
    results=[]
    for t in ticker_list:
        try:
            d=yahoo_fetch(t)
            pe_r=model_pe(d);pb_r=model_pb(d);ev_r=model_ev_ebitda(d);dcf_r=model_dcf(d);gr_r=model_graham(d)
            results.append({"ticker":t,"name":d["name"],"sector":d["sector"],"current_price":d["price"],"currency":d["currency"],"summary":build_summary(d["price"],[pe_r,pb_r,ev_r,dcf_r,gr_r]),"models":{"pe":pe_r,"pb":pb_r,"ev_ebitda":ev_r,"dcf":dcf_r,"graham":gr_r}})
        except Exception as e: results.append({"ticker":t,"error":str(e)})
    return {"comparison":results,"n_stocks":len(results)}
