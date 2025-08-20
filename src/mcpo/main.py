"""
Open WebUI MCPO - main.py v0.0.35o (reconciled to v0.0.29 entrypoint)

Purpose:
- Generate RESTful endpoints from MCP Tool Schemas using the Streamable HTTP MCP client.
- Adds resilience to occasional transient notification/validation noise (e.g., "notifications/initialized")
  surfaced by the HTTP adapter by retrying the RPC once.

Behavior aligned with n8n-mcp (czlonkowski):
- Handshake: initialize -> tools/list -> generate FastAPI routes -> tools/call per invocation.

References:
- n8n MCP/MCPO: https://github.com/DC-AAIA/n8n-mcp
- Railway deploy/logs: https://github.com/DC-AAIA/railwayapp-docs
"""

import os
import json
import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable, Awaitable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_core import ValidationError as PydValidationError
from starlette.responses import JSONResponse

# ----
# MCP client imports (robust across versions)
# ----

from mcp.client.session import ClientSession
from importlib import import_module
from mcp.shared.exceptions import McpError

def resolve_http_connector():
    """
    Resolve a Streamable HTTP MCP connector across known API variants.
    Returns: (connector_callable, connector_name, module_path, mcp_version_str)
    Raises: ImportError with diagnostics if none matched.
    """
    mcp_version = None
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            mcp_version = version("mcp")
        except PackageNotFoundError:
            mcp_version = "unknown"
    except Exception:
        mcp_version = "unknown"

    candidates = []

    # Prefer a connector that directly yields (reader, writer)
    try:
        m = import_module("mcp.client.streamable_http")
        if hasattr(m, "connect"):
            return (m.connect, "streamable_http.connect", getattr(m, "__file__", "<unknown>"), mcp_version)
        if hasattr(m, "connect_streamable_http"):
            return (m.connect_streamable_http, "streamable_http.connect_streamable_http", getattr(m, "__file__", "<unknown>"), mcp_version)
        if hasattr(m, "streamablehttp_client"):
            return (m.streamablehttp_client, "streamable_http.streamablehttp_client", getattr(m, "__file__", "<unknown>"), mcp_version)
        if hasattr(m, "create_mcp_http_client"):
            return (m.create_mcp_http_client, "streamable_http.create_mcp_http_client", getattr(m, "__file__", "<unknown>"), mcp_version)
        candidates.append(("mcp.client.streamable_http", list(sorted(dir(m)))))
    except Exception as e:
        candidates.append(("mcp.client.streamable_http (import error)", str(e)))

    try:
        m = import_module("mcp.client.http.streamable")
        if hasattr(m, "connect"):
            return (m.connect, "http.streamable.connect", getattr(m, "__file__", "<unknown>"), mcp_version)
        candidates.append(("mcp.client.http.streamable", list(sorted(dir(m)))))
    except Exception as e:
        candidates.append(("mcp.client.http.streamable (import error)", str(e)))

    try:
        m = import_module("mcp.client.http")
        if hasattr(m, "connect"):
            return (m.connect, "http.connect", getattr(m, "__file__", "<unknown>"), mcp_version)
        candidates.append(("mcp.client.http", list(sorted(dir(m)))))
    except Exception as e:
        candidates.append(("mcp.client.http (import error)", str(e)))

    details = "; ".join([f"{mod}: {info}" for mod, info in candidates])
    raise ImportError(
        f"No compatible MCP HTTP connector found. Checked streamable_http and http variants. "
        f"Installed mcp version: {mcp_version}. Candidates: {details}"
    )

_CONNECTOR, _CONNECTOR_NAME, _CONNECTOR_MODULE_PATH, _MCP_VERSION = resolve_http_connector()

def _resolve_alt_http_connector():
    try:
        mod = import_module("mcp.client.http.streamable")
        if hasattr(mod, "connect"):
            return getattr(mod, "connect")
    except Exception:
        pass
    try:
        mod = import_module("mcp.client.http")
        if hasattr(mod, "connect"):
            return getattr(mod, "connect")
    except Exception:
        pass
    return None
_ALT_HTTP_CONNECT = _resolve_alt_http_connector()

try:
    _streamable_http_mod = import_module("mcp.client.streamable_http")
    _StreamableHTTPTransport = getattr(_streamable_http_mod, "StreamableHTTPTransport", None)
    _StreamReader = getattr(_streamable_http_mod, "StreamReader", None)
    _StreamWriter = getattr(_streamable_http_mod, "StreamWriter", None)
except Exception:
    _StreamableHTTPTransport = None
    _StreamReader = None
    _StreamWriter = None

try:
    import httpx
except Exception:
    httpx = None

APP_NAME = "Open WebUI MCPO"
APP_VERSION = "0.0.35o"
APP_DESCRIPTION = "Automatically generated API from MCP Tool Schemas"
DEFAULT_PORT = int(os.getenv("PORT", "8080"))
PATH_PREFIX = os.getenv("PATH_PREFIX", "/")
CORS_ALLOWED_ORIGINS = [o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()]
API_KEY = os.getenv("API_KEY", "changeme")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://mcp-streamable-test-production.up.railway.app/mcp")
MCP_HEADERS = os.getenv("MCP_HEADERS", "")

def _parse_headers(hs: str):
    if not hs.strip():
        return None
    pairs = []
    for part in hs.split(","):
        if not part.strip():
            continue
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k:
            pairs.append((k, v))
    return pairs or None

logger = logging.getLogger("mcpo")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class APIKeyHeader(BaseModel):
    api_key: str

def api_dependency():
    from fastapi import Request
    async def _dep(request: Request) -> APIKeyHeader:
        key = request.headers.get("x-api-key")
        if not key or key != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return APIKeyHeader(api_key=key)
    return _dep

class ToolDef(BaseModel):
    name: str
    description: Optional[str] = None
    inputSchema: Dict[str, Any]
    outputSchema: Optional[Dict[str, Any]] = None

async def retry_jsonrpc(call_fn: Callable[[], Awaitable], desc: str, retries: int = 1, sleep_s: float = 0.1):
    for attempt in range(retries + 1):
        try:
            return await call_fn()
        except Exception as e:
            txt = str(e)
            transient = (
                "notifications/initialized" in txt or
                "JSONRPCMessage" in txt or
                "TaskGroup" in txt
            )
            if not transient and hasattr(e, "exceptions"):
                for sub in getattr(e, "exceptions", []):
                    s = str(sub)
                    if "notifications/initialized" in s or "JSONRPCMessage" in s:
                        transient = True
                        break
            if attempt < retries and transient:
                logging.getLogger("mcpo").warning("Transient error on %s; retrying (%d/%d)", desc, attempt + 1, retries)
                await asyncio.sleep(sleep_s)
                continue
            raise

@asynccontextmanager
async def _connector_wrapper(url: str):
    if not _CONNECTOR_NAME.endswith("create_mcp_http_client"):
        async with _CONNECTOR(url) as ctx:
            if isinstance(ctx, tuple) and len(ctx) >= 2:
                yield ctx[0], ctx[1]
            else:
                raise RuntimeError(f"Unexpected connector context result: {type(ctx)}")
        return

    if _StreamableHTTPTransport is None or httpx is None:
        raise RuntimeError("MCP 1.13.0 transport adapter prerequisites missing (StreamableHTTPTransport/httpx)")

    headers = _parse_headers(MCP_HEADERS) or []
    client = httpx.AsyncClient(base_url=url, headers=headers)
    transport = _StreamableHTTPTransport(client)

    def _extract_duplex(t):
        reader = getattr(t, "reader", None)
        writer = getattr(t, "writer", None)

        if reader is None and hasattr(t, "stream_reader"):
            reader = getattr(t, "stream_reader")
        if writer is None and hasattr(t, "stream_writer"):
            writer = getattr(t, "stream_writer")

        if (reader is None or writer is None) and hasattr(t, "duplex"):
            d = getattr(t, "duplex")
            if callable(d):
                try:
                    d = d()
                except Exception:
                    d = None
            if d is not None:
                if reader is None and hasattr(d, "reader"):
                    reader = getattr(d, "reader")
                if writer is None and hasattr(d, "writer"):
                    writer = getattr(d, "writer")

        if (reader is None or writer is None) and hasattr(t, "transport"):
            inner = getattr(t, "transport")
            if inner is not None:
                if reader is None and hasattr(inner, "reader"):
                    reader = getattr(inner, "reader")
                if writer is None and hasattr(inner, "writer"):
                    writer = getattr(inner, "writer")

        if (reader is None or writer is None) and hasattr(t, "get_stream"):
            try:
                pair = t.get_stream()
                if isinstance(pair, tuple) and len(pair) >= 2:
                    reader = reader or pair[0]
                    writer = writer or pair[1]
            except Exception:
                pass

        if reader is None and hasattr(t, "reader") and callable(getattr(t, "reader")):
            try:
                reader = t.reader()
            except Exception:
                pass
        if writer is None and hasattr(t, "writer") and callable(getattr(t, "writer")):
            try:
                writer = t.writer()
            except Exception:
                pass

        return reader, writer

    try:
        r, w = _extract_duplex(transport)
        if r is not None and w is not None:
            try:
                yield r, w
                return
            finally:
                try:
                    await client.aclose()
                except Exception:
                    pass

        if _ALT_HTTP_CONNECT is None:
            raise RuntimeError("StreamableHTTPTransport did not provide stream reader/writer and no alternate HTTP connector is available")

        async with _ALT_HTTP_CONNECT(url) as ctx:
            if isinstance(ctx, tuple) and len(ctx) >= 2:
                yield ctx[0], ctx[1]
            else:
                raise RuntimeError(f"Alternate HTTP connector returned unexpected context: {type(ctx)}")
    finally:
        try:
            await client.aclose()
        except Exception:
            pass

def _safe_get(obj: Any, attr: str, key: str) -> Optional[Any]:
    if obj is None:
        return None
    if hasattr(obj, attr):
        try:
            return getattr(obj, attr)
        except Exception:
            pass
    if isinstance(obj, dict):
        try:
            return obj.get(key)
        except Exception:
            pass
    return None

async def list_mcp_tools(reader, writer) -> List[ToolDef]:
    async with ClientSession(reader, writer) as session:
        # Initialize and safely read protocolVersion across 1.12.x dicts and 1.13.x models
        init_result = await retry_jsonrpc(lambda: session.initialize(), "initialize", retries=1)

        # Additive safety lines bracketing the original proto extraction
        safe_proto = _safe_get(init_result, "protocolVersion", "protocolVersion")
        proto = (init_result or {}).get("protocolVersion")
        proto = safe_proto if safe_proto is not None else proto

        logger.info("Negotiated protocol version: %s", proto)

        # List tools with retry to absorb transient notification/validation noise
        tools_result = await retry_jsonrpc(lambda: session.list_tools(), "tools/list", retries=1)

        # tools_result may be dict-like in 1.12.x or a model with .tools in 1.13.x
        raw_tools: List[Dict[str, Any]] = []
        if isinstance(tools_result, dict):
            raw_tools = tools_result.get("tools", [])
        elif hasattr(tools_result, "tools"):
            raw_tools = getattr(tools_result, "tools") or []

        # Minimal robustness if the first pass was empty
        if not raw_tools and hasattr(tools_result, "tools"):
            try:
                raw_tools = getattr(tools_result, "tools") or []
            except Exception:
                raw_tools = []
        if not raw_tools and isinstance(tools_result, dict):
            raw_tools = tools_result.get("tools", [])

        parsed: List[ToolDef] = []
        for t in raw_tools:
            try:
                parsed.append(
                    ToolDef(
                        name=t["name"],
                        description=t.get("description"),
                        inputSchema=t["inputSchema"],
                        outputSchema=t.get("outputSchema"),
                    )
                )
            except Exception as ex:
                logger.warning("Skipping tool due to schema issue: %s; error: %s", t, ex)
        return parsed

async def call_mcp_tool(reader, writer, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    async with ClientSession(reader, writer) as session:
        try:
            resp = await session.call_tool(name=name, arguments=arguments)
        except McpError as me:
            raise HTTPException(status_code=400, detail={"mcp_error": me.message})
        except PydValidationError as e:
            raise HTTPException(status_code=502, detail={"validation_error": str(e)})

        result = resp or {}
        content = result.get("content") or []
        if not content:
            return {}
        first = content[0]
        text = first.get("text") if isinstance(first, dict) else None
        if not text:
            return {"content": content}
        try:
            return json.loads(text)
        except Exception:
            return {"raw": text}

def create_app() -> FastAPI:
    app = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        description=APP_DESCRIPTION,
        docs_url=f"{PATH_PREFIX.rstrip('/')}/docs" if PATH_PREFIX != "/" else "/docs",
        openapi_url=f"{PATH_PREFIX.rstrip('/')}/openapi.json" if PATH_PREFIX != "/" else "/openapi.json",
    )

    if CORS_ALLOWED_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=CORS_ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get(f"{PATH_PREFIX.rstrip('/')}/health" if PATH_PREFIX != "/" else "/health")
    async def health():
        return {"status": "ok", "name": APP_NAME, "version": APP_VERSION}

    @app.get(f"{PATH_PREFIX.rstrip('/')}/ping" if PATH_PREFIX != "/" else "/ping")
    async def ping():
        return {"pong": True}

    @app.on_event("startup")
    async def on_startup():
        logger.info("Starting MCPO Server...")
        logger.info(" Name: %s", APP_NAME)
        logger.info(" Version: %s", APP_VERSION)
        logger.info(" Description: %s", APP_DESCRIPTION)
        try:
            hostname = os.uname().nodename  # type: ignore[attr-defined]
        except Exception:
            hostname = "unknown"
        logger.info(" Hostname: %s", hostname)
        logger.info(" Port: %s", DEFAULT_PORT)
        logger.info(" API Key: %s", "Provided" if API_KEY else "Not provided")
        logger.info(" CORS Allowed Origins: %s", CORS_ALLOWED_ORIGINS or "[]")
        logger.info(" Path Prefix: %s", PATH_PREFIX)

        logger.info("MCP package version: %s", _MCP_VERSION)
        logger.info("MCP connector resolved: %s (from %s)", _CONNECTOR_NAME, _CONNECTOR_MODULE_PATH)

        logger.info("Echo/Ping routes registered")
        logger.info("Configuring for a single StreamableHTTP MCP Server with URL [%s;]", MCP_SERVER_URL)

        try:
            client_context = _connector_wrapper(MCP_SERVER_URL)
        except Exception as e:
            logger.error("Failed to create MCP client context via %s: %s", _CONNECTOR_NAME, e)
            raise

        async def mount_tool_routes(tools: List[ToolDef]):
            for tool in tools:
                route_path = f"{PATH_PREFIX.rstrip('/')}/tools/{tool.name}" if PATH_PREFIX != "/" else f"/tools/{tool.name}"
                logger.info("Registering route: %s", route_path)

                async def handler(payload: Dict[str, Any], _tool=tool, _route=route_path, dep=Depends(api_dependency())):
                    try:
                        async with _connector_wrapper(MCP_SERVER_URL) as (reader, writer):
                            result = await call_mcp_tool(reader, writer, _tool.name, payload or {})
                        return JSONResponse(status_code=200, content=result)
                    except HTTPException as he:
                        raise he
                    except Exception as e:
                        logger.exception("Error calling tool %s at %s: %s", _tool.name, _route, e)
                        raise HTTPException(status_code=502, detail=str(e))

                app.post(route_path, name=f"tool_{tool.name}", tags=["tools"])(handler)

        async def setup_tools():
            try:
                async with client_context as (reader, writer):
                    tools = await list_mcp_tools(reader, writer)
                if not tools:
                    logger.warning("No tools discovered from MCP server")
                else:
                    logger.info("Discovered %d tool(s) from MCP server", len(tools))
                    await mount_tool_routes(tools)
                return
            except Exception as e:
                logger.warning("Startup tool discovery failed (attempt 1): %s", e, exc_info=True)
                await asyncio.sleep(0.1)

            async with _connector_wrapper(MCP_SERVER_URL) as (reader, writer):
                tools = await list_mcp_tools(reader, writer)
            if not tools:
                logger.warning("No tools discovered from MCP server (after retry)")
            else:
                logger.info("Discovered %d tool(s) from MCP server (after retry)", len(tools))
                await mount_tool_routes(tools)

        try:
            await setup_tools()
        except Exception:
            logger.exception("Error during startup tool discovery/mount")

    @app.get("/")
    async def root():
        return {
            "name": APP_NAME,
            "version": APP_VERSION,
            "description": APP_DESCRIPTION,
            "docs": app.docs_url,
            "openapi": app.openapi_url,
        }

    return app

app = create_app()

def _collect_connector_diagnostics() -> Dict[str, Any]:
    info = {
        "mcp_version": _MCP_VERSION,
        "resolved_connector": _CONNECTOR_NAME,
        "connector_module_path": _CONNECTOR_MODULE_PATH,
        "alt_http_connect_available": bool(_ALT_HTTP_CONNECT is not None),
        "headers_configured": bool(MCP_HEADERS.strip()),
    }
    try:
        _mod = import_module("mcp.client.streamable_http")
        info["streamable_http_module_file"] = getattr(_mod, "__file__", "<unknown>")
        info["streamable_http_attrs"] = sorted([a for a in dir(_mod) if not a.startswith("_")])[:100]
    except Exception as e:
        info["streamable_http_import_error"] = str(e)
    try:
        info["has_StreamableHTTPTransport"] = _StreamableHTTPTransport is not None
        if _StreamableHTTPTransport is not None:
            info["StreamableHTTPTransport_attrs"] = sorted([a for a in dir(_StreamableHTTPTransport) if not a.startswith("_")])[:100]
    except Exception:
        pass
    return info

def attach_mcpo_diagnostics(app: FastAPI) -> None:
    route = f"{PATH_PREFIX.rstrip('/')}/_diagnostic" if PATH_PREFIX != "/" else "/_diagnostic"
    @app.get(route)
    async def _diagnostic(dep=Depends(api_dependency())):
        return {
            "app": {"name": APP_NAME, "version": APP_VERSION, "path_prefix": PATH_PREFIX},
            "mcp": _collect_connector_diagnostics(),
        }

attach_mcpo_diagnostics(app)

def run(host: str = "0.0.0.0", port: int = DEFAULT_PORT, log_level: str = None, reload: bool = False, *args, **kwargs):
    import uvicorn
    uvicorn.run(
        "mcpo.main:app",
        host=host,
        port=port,
        log_level=log_level or os.getenv("UVICORN_LOG_LEVEL", "info"),
        reload=reload,
    )

if __name__ == "__main__":
    run()
