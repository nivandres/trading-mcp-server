"""Data-fetching tools.

Each function returns a JSON-serializable dict. Failures return
{"error": "..."} rather than raising — the agent should handle missing
data gracefully rather than crashing on a network blip.
"""
from __future__ import annotations
import os
import time
import asyncio
import ast
import operator
from datetime import datetime, timedelta, timezone
from typing import Any
import httpx
import yfinance as yf


# ---------------------------------------------------------------------------
# Tiny TTL cache for yfinance / external API calls.
# ---------------------------------------------------------------------------
_CACHE: dict[str, tuple[float, Any]] = {}


def _cache_get(key: str, ttl: int) -> Any | None:
    item = _CACHE.get(key)
    if item is None:
        return None
    ts, value = item
    if time.time() - ts > ttl:
        return None
    return value


def _cache_set(key: str, value: Any) -> None:
    _CACHE[key] = (time.time(), value)


# ---------------------------------------------------------------------------
# Market overview — SPY trend, VIX, sector breadth, regime classification.
# ---------------------------------------------------------------------------
async def fetch_market_overview() -> dict:
    """Returns SPY 50/200 DMA, VIX level, sector 5d perf, regime tag."""
    cached = _cache_get("market_overview", ttl=300)
    if cached:
        return cached

    def _sync():
        spy = yf.Ticker("SPY").history(period="1y", auto_adjust=True)
        vix = yf.Ticker("^VIX").history(period="5d", auto_adjust=True)
        sectors = yf.download(
            ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
             "XLI", "XLU", "IJR", "IWM", "MDY"],
            period="20d", progress=False, auto_adjust=True,
            group_by="ticker", threads=True,
        )
        return spy, vix, sectors

    try:
        spy, vix, sectors = await asyncio.to_thread(_sync)
    except Exception as e:
        return {"error": f"yfinance failed: {e}"}

    if spy.empty or vix.empty:
        return {"error": "Empty SPY or VIX data from yfinance"}

    spy_close = float(spy["Close"].iloc[-1])
    spy_50dma = float(spy["Close"].rolling(50).mean().iloc[-1])
    spy_200dma = float(spy["Close"].rolling(200).mean().iloc[-1])
    trend_bullish = spy_50dma > spy_200dma

    vix_level = float(vix["Close"].iloc[-1])
    vix_high = vix_level > 25

    if not trend_bullish and vix_high:
        regime = "defensive"
    elif not trend_bullish:
        regime = "bear"
    elif vix_high:
        regime = "chop"
    else:
        regime = "bull"

    sector_perf: dict[str, float] = {}
    for ticker in ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
                   "XLI", "XLU", "IJR", "IWM", "MDY"]:
        try:
            col = sectors[ticker]["Close"]
            ret_5d = float(col.iloc[-1] / col.iloc[-6] - 1)
            sector_perf[ticker] = round(ret_5d * 100, 2)
        except (KeyError, IndexError, ValueError):
            continue

    result = {
        "spy_close": round(spy_close, 2),
        "spy_50dma": round(spy_50dma, 2),
        "spy_200dma": round(spy_200dma, 2),
        "trend_bullish": trend_bullish,
        "vix": round(vix_level, 2),
        "vix_high": vix_high,
        "regime": regime,
        "sector_5d_pct": sector_perf,
        "as_of_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _cache_set("market_overview", result)
    return result


# ---------------------------------------------------------------------------
# Stock quotes (current).
# ---------------------------------------------------------------------------
async def fetch_stock_quotes(tickers: list[str]) -> dict:
    """Latest day's OHLCV + day-change for a list of tickers."""
    tickers = [t.upper() for t in tickers if t]
    if not tickers:
        return {"error": "No tickers provided"}

    cache_key = f"quotes:{','.join(sorted(tickers))}"
    cached = _cache_get(cache_key, ttl=60)
    if cached:
        return cached

    def _sync():
        return yf.download(
            tickers, period="5d", interval="1d",
            progress=False, auto_adjust=True, group_by="ticker",
        )

    try:
        df = await asyncio.to_thread(_sync)
    except Exception as e:
        return {"error": f"yfinance failed: {e}"}

    items = []
    for t in tickers:
        try:
            sub = df[t] if len(tickers) > 1 else df
            last = sub["Close"].iloc[-1]
            prev = sub["Close"].iloc[-2] if len(sub) > 1 else last
            items.append({
                "ticker": t,
                "open": round(float(sub["Open"].iloc[-1]), 2),
                "high": round(float(sub["High"].iloc[-1]), 2),
                "low": round(float(sub["Low"].iloc[-1]), 2),
                "close": round(float(last), 2),
                "volume": int(sub["Volume"].iloc[-1]),
                "change_pct": round(float(last / prev - 1) * 100, 2),
            })
        except (KeyError, IndexError, ValueError):
            items.append({"ticker": t, "error": "no data"})

    result = {"items": items, "count": len(items),
              "as_of_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _cache_set(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# Stock candles.
# ---------------------------------------------------------------------------
async def fetch_stock_candles(
    ticker: str, period: str = "1mo", interval: str = "1d", limit: int = 60,
) -> dict:
    """Historical OHLCV bars.

    period   : 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y
    interval : 1m, 5m, 15m, 30m, 60m, 1d, 5d, 1wk
    """
    cache_key = f"candles:{ticker.upper()}:{period}:{interval}:{limit}"
    cached = _cache_get(cache_key, ttl=120)
    if cached:
        return cached

    def _sync():
        return yf.Ticker(ticker.upper()).history(
            period=period, interval=interval, auto_adjust=True,
        )

    try:
        df = await asyncio.to_thread(_sync)
    except Exception as e:
        return {"error": f"yfinance failed: {e}"}

    if df.empty:
        return {"ticker": ticker.upper(), "items": [], "count": 0}

    df = df.tail(limit)
    items = []
    for ts, row in df.iterrows():
        items.append({
            "ts": ts.isoformat(),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        })

    result = {"ticker": ticker.upper(), "period": period,
              "interval": interval, "items": items, "count": len(items)}
    _cache_set(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# Alpaca News (free with paper account).
# ---------------------------------------------------------------------------
async def fetch_news(ticker: str, hours: int = 24, limit: int = 20) -> dict:
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    if not api_key or not secret:
        return {"error": "ALPACA_API_KEY / ALPACA_SECRET_KEY not set", "items": []}

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret,
        "accept": "application/json",
    }
    params = {
        "symbols": ticker.upper(),
        "start": start.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "limit": min(limit, 50),
        "include_content": "true",
        "sort": "desc",
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(
                "https://data.alpaca.markets/v1beta1/news",
                headers=headers, params=params,
            )
    except Exception as e:
        return {"error": f"Alpaca request failed: {e}", "items": []}

    if r.status_code != 200:
        return {"error": f"Alpaca {r.status_code}: {r.text[:200]}", "items": []}

    data = r.json()
    items = []
    for item in data.get("news", []):
        items.append({
            "id": item.get("id"),
            "ts": item.get("created_at"),
            "headline": item.get("headline", "")[:280],
            "summary": (item.get("summary") or "")[:600],
            "source": item.get("source"),
            "author": item.get("author"),
            "url": item.get("url"),
            "symbols": item.get("symbols", []),
        })

    return {"ticker": ticker.upper(), "hours": hours,
            "items": items, "count": len(items)}


# ---------------------------------------------------------------------------
# Finnhub earnings calendar (free).
# ---------------------------------------------------------------------------
async def fetch_earnings_calendar(
    ticker: str = "", days_ahead: int = 7,
) -> dict:
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        return {"error": "FINNHUB_API_KEY not set", "items": []}

    today = datetime.now(timezone.utc).date()
    end = today + timedelta(days=days_ahead)

    params = {
        "from": today.isoformat(),
        "to": end.isoformat(),
        "token": api_key,
    }
    if ticker:
        params["symbol"] = ticker.upper()

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(
                "https://finnhub.io/api/v1/calendar/earnings",
                params=params,
            )
    except Exception as e:
        return {"error": f"Finnhub request failed: {e}", "items": []}

    if r.status_code != 200:
        return {"error": f"Finnhub {r.status_code}: {r.text[:200]}", "items": []}

    data = r.json()
    items = []
    for item in data.get("earningsCalendar", []) or []:
        items.append({
            "ticker": item.get("symbol"),
            "date": item.get("date"),
            "hour": item.get("hour"),
            "eps_estimate": item.get("epsEstimate"),
            "revenue_estimate": item.get("revenueEstimate"),
            "year": item.get("year"),
            "quarter": item.get("quarter"),
        })

    return {"items": items, "count": len(items),
            "from": today.isoformat(), "to": end.isoformat()}


# ---------------------------------------------------------------------------
# Insider transactions via yfinance (parsed Form 4 from Yahoo Finance).
# ---------------------------------------------------------------------------
async def fetch_insider_trading(ticker: str, days: int = 90) -> dict:
    cache_key = f"insider:{ticker.upper()}:{days}"
    cached = _cache_get(cache_key, ttl=3600)
    if cached:
        return cached

    def _sync():
        t = yf.Ticker(ticker.upper())
        try:
            return t.insider_transactions
        except Exception:
            return None

    df = await asyncio.to_thread(_sync)
    if df is None or df.empty:
        return {"ticker": ticker.upper(), "items": [], "count": 0,
                "note": "No insider data via yfinance"}

    items = []
    cutoff = datetime.now() - timedelta(days=days)
    for _, row in df.iterrows():
        date_val = row.get("Start Date") or row.get("Date") or row.get("Most Recent Transaction")
        try:
            date_dt = datetime.fromisoformat(str(date_val))
            if date_dt < cutoff:
                continue
        except Exception:
            pass
        items.append({
            "insider": str(row.get("Insider", "") or ""),
            "title": str(row.get("Position", "") or ""),
            "transaction": str(row.get("Transaction", "") or ""),
            "shares": int(row.get("Shares", 0) or 0),
            "value_usd": float(row.get("Value", 0) or 0),
            "date": str(date_val) if date_val is not None else "",
        })

    buys = [it for it in items if "Buy" in it.get("transaction", "") and it["value_usd"] >= 50_000]
    cluster = len({b["insider"] for b in buys}) >= 3

    result = {
        "ticker": ticker.upper(),
        "items": items[:25],
        "count": len(items),
        "buy_cluster_3plus": cluster,
        "total_buy_value_usd": round(sum(b["value_usd"] for b in buys), 2),
    }
    _cache_set(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# Fundamentals via yfinance.
# ---------------------------------------------------------------------------
async def fetch_fundamentals(ticker: str) -> dict:
    cache_key = f"fund:{ticker.upper()}"
    cached = _cache_get(cache_key, ttl=3600)
    if cached:
        return cached

    def _sync():
        return yf.Ticker(ticker.upper()).info or {}

    try:
        info = await asyncio.to_thread(_sync)
    except Exception as e:
        return {"error": f"yfinance failed: {e}"}

    keys = [
        "shortName", "sector", "industry", "country", "marketCap",
        "enterpriseValue", "trailingPE", "forwardPE", "priceToBook",
        "trailingEps", "forwardEps", "pegRatio", "profitMargins",
        "operatingMargins", "returnOnEquity", "debtToEquity",
        "totalCashPerShare", "totalRevenue", "revenueGrowth",
        "earningsGrowth", "freeCashflow", "beta", "averageVolume",
        "averageDailyVolume10Day", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        "shortRatio", "shortPercentOfFloat",
    ]
    result = {"ticker": ticker.upper()}
    for k in keys:
        if k in info and info[k] is not None:
            result[k] = info[k]
    _cache_set(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# SEC EDGAR Form 8-K recent filings (free).
# ---------------------------------------------------------------------------
async def fetch_recent_filings(ticker: str, form_type: str = "8-K", limit: int = 20) -> dict:
    """Returns recent 8-K (or any form type) filings for a ticker."""
    user_agent = os.getenv("EDGAR_USER_AGENT", "trading-mcp your-email@example.com")

    cik_cache = _cache_get(f"cik:{ticker.upper()}", ttl=86400)
    if cik_cache:
        cik = cik_cache
    else:
        try:
            async with httpx.AsyncClient(timeout=15, headers={"User-Agent": user_agent}) as client:
                r = await client.get("https://www.sec.gov/files/company_tickers.json")
                if r.status_code != 200:
                    return {"error": f"EDGAR ticker map {r.status_code}"}
                tmap = r.json()
        except Exception as e:
            return {"error": f"EDGAR request failed: {e}"}

        cik = None
        for entry in tmap.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                break
        if cik is None:
            return {"ticker": ticker.upper(), "items": [], "count": 0,
                    "note": "Ticker not found in EDGAR map"}
        _cache_set(f"cik:{ticker.upper()}", cik)

    try:
        async with httpx.AsyncClient(timeout=20, headers={"User-Agent": user_agent}) as client:
            r = await client.get(f"https://data.sec.gov/submissions/CIK{cik}.json")
            if r.status_code != 200:
                return {"error": f"EDGAR submissions {r.status_code}"}
            data = r.json()
    except Exception as e:
        return {"error": f"EDGAR submissions failed: {e}"}

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accs = recent.get("accessionNumber", [])
    primary = recent.get("primaryDocument", [])

    items = []
    for form, date, acc, doc in zip(forms, dates, accs, primary):
        if form_type and form != form_type:
            continue
        acc_clean = acc.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{doc}"
        items.append({
            "form": form, "date": date, "accession": acc,
            "url": url, "doc": doc,
        })
        if len(items) >= limit:
            break

    return {"ticker": ticker.upper(), "form_type": form_type,
            "items": items, "count": len(items)}


# ---------------------------------------------------------------------------
# Calculator (safe arithmetic eval).
# ---------------------------------------------------------------------------
_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


async def calculator(expression: str) -> dict:
    """Safe arithmetic evaluator. Supports + - * / // % ** and unary minus."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}
