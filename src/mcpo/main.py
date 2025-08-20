"""
Open WebUI MCPO - main.py v0.0.35g (reconciled to v0.0.29 entrypoint)

Purpose:
- Generate RESTful endpoints from MCP Tool Schemas using the Streamable HTTP MCP client.
- Adds resilience to occasional stray notification bodies (e.g., "notifications/initialized")
  surfaced by the HTTP adapter by retrying the RPC once.

Behavior aligned with n8n-mcp (czlonkowski):
- Handshake: initialize -> tools/list -> generate FastAPI routes -> tools/call per invocation.

References:
- n8n MCP/MCPO: https://github.com/DC-AAIA/n8n-mcp
- Railway deploy/logs: https://github.com/DC-AAIA/railwayapp-docs
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional
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

    # 1) mcp.client.streamable_http.* (primary module)
    try:
        m = import_module("mcp.client.streamable_http")
        # 1a) Oldest variants
        if hasattr(m, "connect"):
            return (m.connect, "streamable_http.connect", getattr(m, "__file__", "<unknown>"), mcp_version)
        # 1b) MCP 1.13.0+ new factory (returns httpx.AsyncClient)
        if hasattr(m, "create_mcp_http_client"):
            return (m.create_mcp_http_client, "streamable_http.create_mcp_http_client", getattr(m, "__file__", "<unknown>"), mcp_version)
        # 1c) Some intermediate variants
        if hasattr(m, "connect_streamable_http"):
            return (m.connect_streamable_http, "streamable_http.connect_streamable_http", getattr(m, "__file__", "<unknown>"), mcp_version)
        candidates.append(("mcp.client.streamable_http", list(sorted(dir(m)))))
    except Exception as e:
        candidates.append(("mcp.client.streamable_http (import error)", str(e)))

    # 2) mcp.client.http.streamable.connect (alt placement)
    try:
        m = import_module("mcp.client.http.streamable")
        if hasattr(m, "connect"):
            return (m.connect, "http.streamable.connect", getattr(m, "__file__", "<unknown>"), mcp_version)
        candidates.append(("mcp.client.http.streamable", list(sorted(dir(m)))))
    except Exception as e:
        candidates.append(("mcp.client.http.streamable (import error)", str(e)))

    # 3) mcp.client.http.connect (generic)
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

# For MCP 1.13.0, we need access to StreamableHTTPTransport and types
try:
    _streamable_http_mod = import_module("mcp.client.streamable_http")
    _StreamableHTTPTransport = getattr(_streamable_http_mod, "StreamableHTTPTransport", None)
    _StreamReader = getattr(_streamable_http_mod, "StreamReader", None)
    _StreamWriter = getattr(_streamable_http_mod, "StreamWriter", None)
except Exception:
    _StreamableHTTPTransport = None
    _StreamReader = None
    _StreamWriter = None

# v0.0.35g: httpx needed for the 1.13.0 adapter branch
try:
    import httpx
except Exception:
    httpx = None

# ----
# App Config
# ----

APP_NAME = "Open WebUI MCPO"
APP_VERSION = "0.0.35g"
APP_DESCRIPTION = "Automatically generated API from MCP Tool Schemas"
DEFAULT_PORT = int(os.getenv("PORT", "8080"))
PATH_PREFIX = os.getenv("PATH_PREFIX", "/")
CORS_ALLOWED_ORIGINS = [o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()]
API_KEY = os.getenv("API_KEY", "changeme")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://mcp-streamable-test-production.up.railway.app/mcp")

# Optional headers for MCP HTTP client (ensure sequence of (k, v) tuples if set)
MCP_HEADERS = os.getenv("MCP_HEADERS", "")  # format: "Key1:Val1,Key2:Val2"
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

# ----
# Logging
# ----

logger = logging.getLogger("mcpo")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ----
# Security dependency
# ----

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

# ----
# Models mirroring MCP tool schemas
# ----

class ToolDef(BaseModel):
    name: str
    description: Optional[str] = None
    inputSchema: Dict[str, Any]
    outputSchema: Optional[Dict[str, Any]] = None

# ----
# Resilience helper: skip stray notification envelopes and retry once
# ----

async def rpc_with_skip_notifications(coro, desc: str):
    """
    Runs an MCP RPC coroutine. If the streamable HTTP adapter surfaces a stray
    notification body (e.g., 'notifications/initialized') as the HTTP response
    and the MCP client raises a Pydantic validation error, skip once and retry.
    """
    try:
        return await coro
    except PydValidationError as e:
        txt = str(e)
        if "notifications/initialized" in txt or "JSONRPCMessage" in txt:
            logger.warning("rpc_with_skip_notifications: validation issue on %s; retrying once", desc)
            return await coro
        raise

# ----
# Connector wrapper for MCP 1.13.0 (normalize to yield (reader, writer))
# ----

@asynccontextmanager
async def _connector_wrapper(url: str):
    """
    Normalizes connectors to an async context that yields (reader, writer).
    - For 'create_mcp_http_client', build StreamableHTTPTransport around the httpx.AsyncClient.
    - For legacy connectors returning (reader, writer), simply delegate.
    """
    if _CONNECTOR_NAME.endswith("create_mcp_http_client"):
        if _StreamableHTTPTransport is None:
            raise RuntimeError("StreamableHTTPTransport not available in mcp.client.streamable_http")
        if httpx is None:
            raise RuntimeError("httpx is required for MCP 1.13.0 transport adapter")

        # Build an httpx.AsyncClient using base_url + headers
        headers = _parse_headers(MCP_HEADERS) or []
        client = httpx.AsyncClient(base_url=url, headers=headers)

        transport = _StreamableHTTPTransport(client)

        # Probe common attribute names for the duplex stream; be robust across MCP variants.
        try:
            reader = getattr(transport, "reader", None)
            writer = getattr(transport, "writer", None)

            if reader is None and hasattr(transport, "stream_reader"):
                reader = getattr(transport, "stream_reader")
            if writer is None and hasattr(transport, "stream_writer"):
                writer = getattr(transport, "stream_writer")

            # Some builds expose a duplex object with .reader/.writer
            if (reader is None or writer is None) and hasattr(transport, "duplex"):
                duplex = getattr(transport, "duplex")
                if duplex is not None:
                    if reader is None and hasattr(duplex, "reader"):
                        reader = getattr(duplex, "reader")
                    if writer is None and hasattr(duplex, "writer"):
                        writer = getattr(duplex, "writer")

            # Some variants offer a method returning both ends
            if (reader is None or writer is None) and hasattr(transport, "get_stream"):
                try:
                    pair = transport.get_stream()
                    if isinstance(pair, tuple) and len(pair) >= 2:
                        reader = reader or pair[0]
                        writer = writer or pair[1]
                except Exception:
                    pass

            # Last resort: callables reader()/writer()
            if reader is None and hasattr(transport, "reader") and callable(getattr(transport, "reader")):
                try:
                    reader = transport.reader()
                except Exception:
                    pass
            if writer is None and hasattr(transport, "writer") and callable(getattr(transport, "writer")):
                try:
                    writer = transport.writer()
                except Exception:
                    pass

            if reader is None or writer is None:
                raise RuntimeError("StreamableHTTPTransport did not provide stream reader/writer")

            yield reader, writer
        finally:
            try:
                await client.aclose()
            except Exception:
                pass
    else:
        # Legacy or alternative connectors already yield (reader, writer) or (reader, writer, ...)
        async with _CONNECTOR(url) as ctx:
            if isinstance(ctx, tuple) and len(ctx) >= 2:
                yield ctx[0], ctx[1]
            else:
                raise RuntimeError(f"Unexpected connector context result: {type(ctx)}")

# ----
# MCP client lifecycle and dynamic route generation
# ----

async def list_mcp_tools(reader, writer) -> List[ToolDef]:
    async with ClientSession(reader, writer) as session:
        init_result = await rpc_with_skip_notifications(session.initialize(), "initialize")
        proto = (init_result or {}).get("protocolVersion")
        logger.info("Negotiated protocol version: %s", proto)

        tools_result = await rpc_with_skip_notifications(session.list_tools(), "tools/list")
        tools = tools_result.get("tools", [])
        parsed: List[ToolDef] = []
        for t in tools:
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
        text = content[0].get("text") if isinstance(content[0], dict) else None
        if not text:
            return {"content": content}
        try:
            return json.loads(text)
        except Exception:
            return {"raw": text}

# ----
# FastAPI app and dynamic route creation
# ----

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
        logger.info(" Hostname: %s", os.uname().nodename if hasattr(os, "uname") else "unknown")
        logger.info(" Port: %s", DEFAULT_PORT)
        logger.info(" API Key: %s", "Provided" if API_KEY else "Not provided")
        logger.info(" CORS Allowed Origins: %s", CORS_ALLOWED_ORIGINS or "[]")
        logger.info(" Path Prefix: %s", PATH_PREFIX)

        # Prove which MCP lib we resolved and where it came from
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
            async with client_context as (reader, writer):
                tools = await list_mcp_tools(reader, writer)
                if not tools:
                    logger.warning("No tools discovered from MCP server")
                else:
                    logger.info("Discovered %d tool(s) from MCP server", len(tools))
                await mount_tool_routes(tools)

        try:
            await setup_tools()
        except Exception as e:
            logger.error("Error during startup tool discovery/mount: %s", e)

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

# ----
# Backward-compatible entrypoint expected by container (v0.0.29-compatible signature)
# ----

def run(host: str = "0.0.0.0", port: int = DEFAULT_PORT, log_level: str = None, reload: bool = False, *args, **kwargs):
    """
    Backward-compatible entrypoint: start uvicorn server.
    Accepts keyword arguments because some Railway entrypoints call run(host=..., port=..., log_level=...).
    """
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
