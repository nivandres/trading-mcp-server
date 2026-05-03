"""SSE-based MCP server exposing trading data tools.

Endpoints:
    GET  /sse        — SSE stream (MCP transport, server.pyrequires Bearer auth)
    POST /messages/  — MCP message channel (Bearserver.pyer auth)
    GET  /health     — unauth health check (for Render / uptime monitors)

Run locally:
    uvicorn server:app --host 0.0.0.0 --port 8000

Deploy on Render:
    See README.md and render.yaml.
"""
from __future__ import annotations
import json
import logging
import os
from typing import Any
from dotenv import load_dotenv

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, Mount

import tools

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("trading-mcp")


TOOLS: dict[str, dict[str, Any]] = {
    "fetch_market_overview": {
        "handler": tools.fetch_market_overview,
        "description": (
            "Get current market regime: SPY 50/200 DMA, VIX level, sector ETF "
            "5-day momentum, and computed regime tag (bull/chop/bear/defensive). "
            "Cached 5 min. No arguments."
        ),
        "schema": {"type": "object", "properties": {}, "required": []},
    },
    "fetch_stock_quotes": {
        "handler": tools.fetch_stock_quotes,
        "description": "Get latest day's OHLCV + day-change for one or more US tickers. Cached 1 min.",
        "schema": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}, "description": "List of ticker symbols, e.g. ['AAPL','MSFT']."},
            },
            "required": ["tickers"],
        },
    },
    "fetch_stock_candles": {
        "handler": tools.fetch_stock_candles,
        "description": "Historical OHLCV bars for one ticker. Cached 2 min.",
        "schema": {
            "type": "object",
            "properties": {
                "ticker":   {"type": "string"},
                "period":   {"type": "string", "description": "1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y", "default": "1mo"},
                "interval": {"type": "string", "description": "1m, 5m, 15m, 30m, 60m, 1d, 5d, 1wk", "default": "1d"},
                "limit":    {"type": "integer", "default": 60},
            },
            "required": ["ticker"],
        },
    },
    "fetch_news": {
        "handler": tools.fetch_news,
        "description": "Recent news articles for a ticker via Alpaca News (Benzinga-sourced).",
        "schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "hours":  {"type": "integer", "default": 24},
                "limit":  {"type": "integer", "default": 20},
            },
            "required": ["ticker"],
        },
    },
    "fetch_earnings_calendar": {
        "handler": tools.fetch_earnings_calendar,
        "description": "Upcoming earnings via Finnhub.",
        "schema": {
            "type": "object",
            "properties": {
                "ticker":     {"type": "string", "default": ""},
                "days_ahead": {"type": "integer", "default": 7},
            },
            "required": [],
        },
    },
    "fetch_insider_trading": {
        "handler": tools.fetch_insider_trading,
        "description": "Recent insider transactions (Form 4) for a ticker.",
        "schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "days":   {"type": "integer", "default": 90},
            },
            "required": ["ticker"],
        },
    },
    "fetch_fundamentals": {
        "handler": tools.fetch_fundamentals,
        "description": "Key fundamentals for a ticker: market cap, P/E, margins, growth, debt/equity, 52w range.",
        "schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
    "fetch_recent_filings": {
        "handler": tools.fetch_recent_filings,
        "description": "Recent SEC filings for a ticker (default 8-K).",
        "schema": {
            "type": "object",
            "properties": {
                "ticker":    {"type": "string"},
                "form_type": {"type": "string", "default": "8-K"},
                "limit":     {"type": "integer", "default": 20},
            },
            "required": ["ticker"],
        },
    },
    "calculator": {
        "handler": tools.calculator,
        "description": "Safe arithmetic evaluator. Supports + - * / // % ** and unary minus.",
        "schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}

mcp = Server("trading-data-mcp")

@mcp.list_tools()
async def list_tools() -> list[Tool]:
    return [Tool(name=name, description=spec["description"], inputSchema=spec["schema"]) for name, spec in TOOLS.items()]

@mcp.call_tool()
async def call_tool(name: str, arguments: dict | None) -> list[TextContent]:
    spec = TOOLS.get(name)
    if spec is None:
        result = {"error": f"Unknown tool: {name}"}
    else:
        try:
            result = await spec["handler"](**(arguments or {}))
        except TypeError as e:
            result = {"error": f"Bad arguments for {name}: {e}"}
        except Exception as e:
            log.exception("Tool %s failed", name)
            result = {"error": f"{name} crashed: {e}"}
    return [TextContent(type="text", text=json.dumps(result, default=str))]

sse_transport = SseServerTransport("/messages/")

async def handle_sse(request):
    async with sse_transport.connect_sse(request.scope, request.receive, request._send) as (read_stream, write_stream):
        await mcp.run(read_stream, write_stream, InitializationOptions(
            server_name="trading-data-mcp", server_version="0.1.0",
            capabilities=mcp.get_capabilities(notification_options=NotificationOptions(), experimental_capabilities={}),
        ))

async def health(_request):
    configured = []
    if os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"):
        configured.append("alpaca_news")
    if os.getenv("FINNHUB_API_KEY"):
        configured.append("finnhub_calendar")
    configured += ["yfinance", "edgar_filings"]
    return JSONResponse({"status": "ok", "service": "trading-mcp", "configured_sources": configured, "tool_count": len(TOOLS)})

class BearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, expected_token: str, public_paths=("/health",)):
        super().__init__(app)
        self.expected_token = expected_token
        self.public_paths = public_paths

    async def dispatch(self, request, call_next):
        if any(request.url.path == p or request.url.path.startswith(p + "/") for p in self.public_paths):
            return await call_next(request)
        if not self.expected_token:
            return JSONResponse({"error": "Server not configured: MCP_BEARER_TOKEN missing"}, status_code=503)
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse({"error": "missing bearer token"}, status_code=401)
        if auth[7:].strip() != self.expected_token:
            return JSONResponse({"error": "invalid bearer token"}, status_code=401)
        return await call_next(request)

log.info("trading-mcp starting up — tools: %s", list(TOOLS.keys()))
if not os.getenv("MCP_BEARER_TOKEN"):
    log.warning("MCP_BEARER_TOKEN not set")

app = Starlette(routes=[
    Route("/health", endpoint=health, methods=["GET"]),
    Route("/sse", endpoint=handle_sse),
    Mount("/messages/", app=sse_transport.handle_post_message),
])
app.add_middleware(BearerAuthMiddleware, expected_token=os.getenv("MCP_BEARER_TOKEN", ""))
