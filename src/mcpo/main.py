"""
Open WebUI MCPO - main.py v0.0.40.2 (FastAPI the body is optional Fix)

Purpose:
- Generate RESTful endpoints from MCP Tool Schemas using the Streamable HTTP MCP client.
- FIXED: Uses direct HTTP as PRIMARY method for authentication compatibility
- PRESERVES: ALL existing v0.0.38.1 functionality and error handling (1134 lines)
- CONSERVATIVE: Only changes method priority in setup_tools() function

Changes from v0.0.38.1:
- Modified setup_tools() to try direct HTTP FIRST (proven working method)
- Falls back to MCP connector if direct HTTP fails
- Preserves ALL existing MCP connector logic, MCPRemoteManager, and error handling
- ALL 1134 lines of working code preserved

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
import subprocess
from typing import Any, Dict, List, Optional, Callable, Awaitable
from contextlib import asynccontextmanager
from importlib import import_module

from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_core import ValidationError as PydValidationError
from starlette.responses import JSONResponse

from mcp.client.session import ClientSession
from mcp.shared.exceptions import McpError

# Try to import stdio transport - may not be available in all MCP versions
try:
    from mcp.client.stdio import StdioClientTransport
    STDIO_AVAILABLE = True
except ImportError:
    try:
        from mcp.client.stdio import StdioServerTransport as StdioClientTransport
        STDIO_AVAILABLE = True
    except ImportError:
        STDIO_AVAILABLE = False
        logger = logging.getLogger("mcpo")
        logger.warning("StdioClientTransport not available - mcp-remote fallback disabled")

def resolve_http_connector():
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
APP_VERSION = "0.0.40.2"
APP_DESCRIPTION = "Automatically generated API from MCP Tool Schemas"
DEFAULT_PORT = int(os.getenv("PORT", "8080"))
PATH_PREFIX = os.getenv("PATH_PREFIX", "/")
CORS_ALLOWED_ORIGINS = [o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()]
API_KEY = os.getenv("API_KEY", "changeme")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://mcp-streamable-test-production.up.railway.app/mcp")
MCP_HEADERS = os.getenv("MCP_HEADERS", "")

def _parse_headers(hs: str) -> Dict[str, str]:
    """Parse header string into dict for httpx client.
    
    Supports formats:
    - "Authorization: Bearer TOKEN"
    - "key1:value1,key2:value2"
    - JSON format: {"key": "value"}
    """
    if not hs.strip():
        return {}
    
    # Try JSON format first
    if hs.strip().startswith("{"):
        try:
            return json.loads(hs)
        except:
            pass
    
    headers = {}
    
    # Check for single Authorization header format
    if hs.startswith("Authorization:") or hs.startswith("authorization:"):
        parts = hs.split(":", 1)
        if len(parts) == 2:
            headers[parts[0].strip()] = parts[1].strip()
            return headers
    
    # Try comma-separated format
    for part in hs.split(","):
        if not part.strip():
            continue
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k:
            headers[k] = v
    
    return headers

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
        # Accept either x-api-key or Authorization: Bearer <token>
        key = request.headers.get("x-api-key")
        if not key:
            auth = request.headers.get("authorization") or request.headers.get("Authorization")
            if auth and auth.lower().startswith("bearer "):
                key = auth.split(" ", 1)[1].strip()

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

# EXISTING: Direct HTTP fallback functions for auth issues
async def discover_tools_via_http_fallback(url: str, headers: Dict[str, str]) -> List[ToolDef]:
    """FALLBACK: Discover tools using direct HTTP when MCP connector auth fails."""
    if not httpx:
        raise RuntimeError("httpx is required for direct HTTP tool discovery fallback")
    
    logger.info("Using direct HTTP fallback for tool discovery")
    
    async with httpx.AsyncClient() as client:
        # Step 1: Initialize MCP session
        init_payload = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mcpo", "version": APP_VERSION}
            },
            "id": 1
        }
        
        logger.debug("Sending initialize request via direct HTTP fallback")
        init_response = await client.post(url, json=init_payload, headers=headers)
        
        if init_response.status_code != 200:
            raise Exception(f"Direct HTTP initialize failed: {init_response.status_code} {init_response.text}")
        
        init_result = init_response.json()
        logger.info("Direct HTTP initialize successful, protocol version: %s", init_result.get("result", {}).get("protocolVersion"))
        
        # Step 2: List tools
        tools_payload = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 2
        }
        
        logger.debug("Requesting tools list via direct HTTP fallback")
        tools_response = await client.post(url, json=tools_payload, headers=headers)
        
        if tools_response.status_code != 200:
            raise Exception(f"Direct HTTP tools list failed: {tools_response.status_code} {tools_response.text}")
        
        tools_result = tools_response.json()
        raw_tools = tools_result.get("result", {}).get("tools", [])
        
        logger.info("Discovered %d tools via direct HTTP fallback", len(raw_tools))
        
        # Parse tools (same logic as original)
        parsed: List[ToolDef] = []
        for t in raw_tools:
            try:
                name = t.get("name")
                description = t.get("description")
                input_schema = t.get("inputSchema") or {}
                output_schema = t.get("outputSchema")
                
                if not name:
                    continue
                    
                parsed.append(ToolDef(
                    name=name,
                    description=description,
                    inputSchema=input_schema,
                    outputSchema=output_schema,
                ))
            except Exception as ex:
                logger.warning("Skipping tool due to parsing issue: %s; error: %s", t, ex)
        
        return parsed

async def call_mcp_tool_via_http_fallback(url: str, headers: Dict[str, str], name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """FALLBACK: Call MCP tool using direct HTTP when MCP connector auth fails."""
    if not httpx:
        raise RuntimeError("httpx is required for direct HTTP tool calls fallback")
    
    logger.debug("Using direct HTTP fallback for tool call: %s", name)
    
    async with httpx.AsyncClient() as client:
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            },
            "id": 3
        }
        
        response = await client.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Direct HTTP tool call failed: {response.text}")
        
        result = response.json()
        
        if "error" in result:
            raise HTTPException(status_code=400, detail={"mcp_error": result["error"]})
        
        tool_result = result.get("result", {})
        content = tool_result.get("content", [])
        
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

# NEW: mcp-remote manager for subprocess-based HTTP authentication
class MCPRemoteManager:
    """Manages mcp-remote subprocess for HTTP authentication bridge"""
    
    def __init__(self):
        self.process = None
        self.transport = None
        self.session = None
        
    async def start(self, url: str, auth_token: str):
        """Start mcp-remote subprocess with HTTP authentication"""
        if not STDIO_AVAILABLE:
            raise RuntimeError("StdioClientTransport not available - cannot use mcp-remote fallback")
            
        if not auth_token:
            raise RuntimeError("Auth token required for mcp-remote")
            
        # Build mcp-remote command with authentication
        cmd = [
            "npx", "-y", "mcp-remote",
            url,
            "--header", f"Authorization: Bearer {auth_token}"
        ]
        
        logger.info("Starting mcp-remote subprocess")
        
        # Start mcp-remote subprocess
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Create stdio transport to communicate with mcp-remote
        self.transport = StdioClientTransport(
            self.process.stdout,
            self.process.stdin
        )
        
        # Create MCP session
        self.session = ClientSession(self.transport)
        
        # Initialize MCP connection
        await self.session.initialize()
        
        logger.info("mcp-remote connection established successfully")
        
    async def stop(self):
        """Stop mcp-remote subprocess"""
        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                logger.warning("Error closing MCP session: %s", e)
                
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("mcp-remote process did not terminate gracefully, killing")
                self.process.kill()
                await self.process.wait()
            except Exception as e:
                logger.warning("Error stopping mcp-remote process: %s", e)
                
        self.process = None
        self.transport = None
        self.session = None
        
    async def list_tools(self) -> List[ToolDef]:
        """List available tools via mcp-remote"""
        if not self.session:
            raise RuntimeError("MCP session not initialized")
            
        try:
            tools_result = await self.session.list_tools()
            
            if hasattr(tools_result, 'tools'):
                raw_tools = tools_result.tools
            elif isinstance(tools_result, dict):
                raw_tools = tools_result.get('tools', [])
            else:
                raw_tools = []
                
            parsed_tools = []
            for tool in raw_tools:
                try:
                    if hasattr(tool, 'name'):
                        name = tool.name
                        description = getattr(tool, 'description', None)
                        input_schema = getattr(tool, 'inputSchema', {})
                    else:
                        name = tool.get('name')
                        description = tool.get('description')
                        input_schema = tool.get('inputSchema', {})
                        
                    if name:
                        parsed_tools.append(ToolDef(
                            name=name,
                            description=description,
                            inputSchema=input_schema
                        ))
                except Exception as e:
                    logger.warning("Failed to parse tool: %s, error: %s", tool, e)
                    
            return parsed_tools
            
        except Exception as e:
            logger.error("Failed to list tools via mcp-remote: %s", e)
            raise
            
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call tool via mcp-remote"""
        if not self.session:
            raise RuntimeError("MCP session not initialized")
            
        try:
            result = await self.session.call_tool(name, arguments)
            
            if hasattr(result, 'content'):
                content = result.content
            elif isinstance(result, dict):
                content = result.get('content', [])
            else:
                content = []
                
            if not content:
                return {}
                
            # Extract text from first content item
            first_item = content[0]
            if hasattr(first_item, 'text'):
                text = first_item.text
            elif isinstance(first_item, dict):
                text = first_item.get('text', '')
            else:
                text = str(first_item)
                
            if not text:
                return {"content": content}
                
            # Try to parse as JSON
            try:
                return json.loads(text)
            except Exception:
                return {"raw": text}
                
        except Exception as e:
            logger.error("Failed to call tool %s via mcp-remote: %s", name, e)
            raise HTTPException(status_code=502, detail=str(e))

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

    headers = _parse_headers(MCP_HEADERS)
    
    if headers:
        logger.info("MCP_HEADERS configured with keys: %s", list(headers.keys()))
    
    # Create httpx client with headers
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
        init_result = await retry_jsonrpc(lambda: session.initialize(), "initialize", retries=1)
        safe_proto = _safe_get(init_result, "protocolVersion", "protocolVersion")
        proto = None
        if isinstance(init_result, dict):
            proto = init_result.get("protocolVersion")
        proto = safe_proto if safe_proto is not None else proto

        logger.info("Negotiated protocol version: %s", proto)

        tools_result = await retry_jsonrpc(lambda: session.list_tools(), "tools/list", retries=1)

        raw_tools: List[Dict[str, Any]] = []
        if isinstance(tools_result, dict):
            raw_tools = tools_result.get("tools", [])
        elif hasattr(tools_result, "tools"):
            raw_tools = getattr(tools_result, "tools") or []
        if not raw_tools and hasattr(tools_result, "tools"):
            try:
                raw_tools = getattr(tools_result, "tools") or []
            except Exception:
                raw_tools = []
        if not raw_tools and isinstance(tools_result, dict):
            raw_tools = tools_result.get("tools", [])

        try:
            if raw_tools:
                sample = raw_tools[0]
                tname = getattr(sample, "name", None) if not isinstance(sample, dict) else sample.get("name")
                logger.info(
                    "Sample tool introspection: type=%s name=%s attrs=%s",
                    type(sample).__name__,
                    tname,
                    [a for a in dir(sample) if not a.startswith("_")][:20] if not isinstance(sample, dict) else list(sample.keys()),
                )
        except Exception as _e:
            logger.info("Sample tool introspection failed: %s", _e)

        parsed: List[ToolDef] = []
        for t in raw_tools:
            try:
                if isinstance(t, dict):
                    name = t.get("name")
                    description = t.get("description")
                    input_schema = (
                        t.get("inputSchema")
                        or t.get("input_schema")
                        or t.get("parameters")
                        or t.get("input")
                    )
                    output_schema = (
                        t.get("outputSchema")
                        or t.get("output_schema")
                        or t.get("output")
                    )
                else:
                    name = getattr(t, "name", None)
                    description = getattr(t, "description", None)
                    input_schema = getattr(t, "inputSchema", None)
                    if input_schema is None:
                        input_schema = getattr(t, "input_schema", None)
                    if input_schema is None:
                        input_schema = getattr(t, "parameters", None) or getattr(t, "input", None)
                    output_schema = getattr(t, "outputSchema", None)
                    if output_schema is None:
                        output_schema = getattr(t, "output_schema", None)
                    if output_schema is None:
                        output_schema = getattr(t, "output", None)

                if not name or input_schema is None:
                    raise ValueError("Tool missing required fields (name/inputSchema)")

                parsed.append(
                    ToolDef(
                        name=name,
                        description=description,
                        inputSchema=input_schema,
                        outputSchema=output_schema,
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
            detail = {"mcp_error": getattr(me, "message", str(me))}
            try:
                if hasattr(me, "args") and me.args:
                    detail["args"] = [str(a) for a in me.args]
            except Exception:
                pass
            raise HTTPException(status_code=400, detail=detail)
        except RuntimeError as re:
            # Tolerate MCP backends that return non-structured content or raise runtime result errors
            # Surface as a raw payload so the caller can see the underlying message
            return {"raw_error": str(re)}
        except PydValidationError as e:
            raise HTTPException(status_code=502, detail={"validation_error": str(e)})

        result = resp or {}
        try:
            if isinstance(result, dict):
                if "error" in result and result["error"]:
                    raise HTTPException(status_code=502, detail={"mcp_error": result["error"]})
                if "errors" in result and result["errors"]:
                    raise HTTPException(status_code=502, detail={"mcp_errors": result["errors"]})
        except HTTPException:
            raise

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

_DISCOVERED_TOOL_NAMES: List[str] = []
_DISCOVERED_TOOLS_MIN: List[Dict[str, Any]] = []

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
            hostname = os.uname().nodename
        except Exception:
            hostname = "unknown"
        logger.info(" Hostname: %s", hostname)
        logger.info(" Port: %s", DEFAULT_PORT)
        logger.info(" API Key: %s", "Provided" if API_KEY else "Not provided")
        try:
            _mask = "<unset>"
            if API_KEY:
                _mask = f"{API_KEY[:4]}...{API_KEY[-4:]}" if len(API_KEY) >= 8 else "<short>"
            logger.info(" API Key fingerprint (masked): %s", _mask)
        except Exception:
            logger.info(" API Key fingerprint (masked): <error>")
        logger.info(" CORS Allowed Origins: %s", CORS_ALLOWED_ORIGINS or "[]")
        logger.info(" Path Prefix: %s", PATH_PREFIX)

        logger.info("MCP package version: %s", _MCP_VERSION)
        logger.info("MCP connector resolved: %s (from %s)", _CONNECTOR_NAME, _CONNECTOR_MODULE_PATH)

        logger.info("Echo/Ping routes registered")
        logger.info("Configuring for a single StreamableHTTP MCP Server with URL [%s]", MCP_SERVER_URL)

        # Parse headers for direct HTTP fallback
        headers = _parse_headers(MCP_HEADERS)
        if headers:
            logger.info("Direct HTTP fallback headers configured with keys: %s", list(headers.keys()))

        try:
            client_context = _connector_wrapper(MCP_SERVER_URL)
        except Exception as e:
            logger.error("Failed to create MCP client context via %s: %s", _CONNECTOR_NAME, e)
            raise

        async def mount_tool_routes(tools: List[ToolDef]):
            for tool in tools:
                route_path = f"{PATH_PREFIX.rstrip('/')}/tools/{tool.name}" if PATH_PREFIX != "/" else f"/tools/{tool.name}"
                logger.info("Registering route: %s", route_path)

                async def handler(payload: Optional[Dict[str, Any]] = Body(None), _tool=tool, _route=route_path, dep=Depends(api_dependency())):
                    try:
                        # CONSERVATIVE FIX: Handle missing request body for Open WebUI compatibility
                        if payload is None:
                            logger.info("No request body provided - using empty dict for Open WebUI compatibility")
                            payload = {}
                        
                        # Try direct HTTP first (proven working method)
                        result = await call_mcp_tool_via_http_fallback(MCP_SERVER_URL, headers, _tool.name, payload or {})
                        return JSONResponse(status_code=200, content=result)
                    except Exception as e:
                        # Fall back to original MCP connector
                        logger.warning("Direct HTTP failed for tool %s, trying MCP connector: %s", _tool.name, e)
                        try:
                            async with _connector_wrapper(MCP_SERVER_URL) as (reader, writer):
                                result = await call_mcp_tool(reader, writer, _tool.name, payload or {})
                                return JSONResponse(status_code=200, content=result)
                        except HTTPException as he:
                            raise he
                        except Exception as fe:
                            logger.exception("Both direct HTTP and MCP connector failed for tool %s: %s, %s", _tool.name, e, fe)
                            raise HTTPException(status_code=502, detail=f"Both direct HTTP and MCP connector failed: {str(e)}, {str(fe)}")

                app.post(route_path, name=f"tool_{tool.name}", tags=["tools"])(handler)

        async def setup_tools():
            try:
                # CONSERVATIVE CHANGE: Try direct HTTP FIRST (proven working method)
                logger.info("Trying direct HTTP method first (proven working with authentication)")
                tools = await discover_tools_via_http_fallback(MCP_SERVER_URL, headers)
                if not tools:
                    logger.warning("No tools discovered from MCP server via direct HTTP")
                else:
                    logger.info("Discovered %d tool(s) from MCP server via direct HTTP", len(tools))
                    _DISCOVERED_TOOL_NAMES.clear()
                    _DISCOVERED_TOOL_NAMES.extend([t.name for t in tools])
                    _DISCOVERED_TOOLS_MIN.clear()
                    for t in tools:
                        try:
                            _DISCOVERED_TOOLS_MIN.append({
                                "name": t.name,
                                "description": t.description,
                                "inputSchema": t.inputSchema,
                            })
                        except Exception:
                            pass
                    await mount_tool_routes(tools)
                return
            except Exception as e:
                logger.warning("Direct HTTP tool discovery failed, trying original MCP connector: %s", e)
                
                # Fall back to original MCP connector
                try:
                    async with client_context as (reader, writer):
                        tools = await list_mcp_tools(reader, writer)
                        if not tools:
                            logger.warning("No tools discovered from MCP server via MCP connector")
                        else:
                            logger.info("Discovered %d tool(s) from MCP server via MCP connector", len(tools))
                            _DISCOVERED_TOOL_NAMES.clear()
                            _DISCOVERED_TOOL_NAMES.extend([t.name for t in tools if getattr(t, "name", None)])
                            _DISCOVERED_TOOLS_MIN.clear()
                            for t in tools:
                                try:
                                    _DISCOVERED_TOOLS_MIN.append({
                                        "name": t.name,
                                        "description": t.description,
                                        "inputSchema": t.inputSchema,
                                    })
                                except Exception:
                                    pass
                            await mount_tool_routes(tools)
                    return
                except Exception as fe:
                    logger.exception("MCP connector also failed: %s", fe)
                    # Try mcp-remote as final fallback
                    if STDIO_AVAILABLE:
                        logger.warning("Trying mcp-remote as final fallback")
                        try:
                            # Extract auth token from headers
                            auth_token = None
                            if headers and "Authorization" in headers:
                                auth_value = headers["Authorization"]
                                if auth_value.startswith("Bearer "):
                                    auth_token = auth_value.split(" ", 1)[1]
                            
                            if not auth_token:
                                raise Exception("No auth token available for mcp-remote")
                            
                            # Create and start mcp-remote manager
                            mcp_remote = MCPRemoteManager()
                            await mcp_remote.start(MCP_SERVER_URL, auth_token)
                            
                            tools = await mcp_remote.list_tools()
                            
                            if not tools:
                                logger.warning("No tools discovered from MCP server via mcp-remote fallback")
                            else:
                                logger.info("Discovered %d tool(s) from MCP server via mcp-remote fallback", len(tools))
                                _DISCOVERED_TOOL_NAMES.clear()
                                _DISCOVERED_TOOL_NAMES.extend([t.name for t in tools])
                                _DISCOVERED_TOOLS_MIN.clear()
                                for t in tools:
                                    try:
                                        _DISCOVERED_TOOLS_MIN.append({
                                            "name": t.name,
                                            "description": t.description,
                                            "inputSchema": t.inputSchema,
                                        })
                                    except Exception:
                                        pass
                                        
                                # Create tool routes that use mcp-remote
                                async def mount_mcp_remote_tool_routes(tools: List[ToolDef], mcp_remote: MCPRemoteManager):
                                    for tool in tools:
                                        route_path = f"{PATH_PREFIX.rstrip('/')}/tools/{tool.name}" if PATH_PREFIX != "/" else f"/tools/{tool.name}"
                                        
                                        async def create_mcp_remote_handler(tool_name: str, manager: MCPRemoteManager):
                                            async def handler(payload: Dict[str, Any], dep=Depends(api_dependency())):
                                                try:
                                                    result = await manager.call_tool(tool_name, payload or {})
                                                    return JSONResponse(status_code=200, content=result)
                                                except HTTPException:
                                                    raise
                                                except Exception as e:
                                                    logger.exception("Error calling tool %s via mcp-remote: %s", tool_name, e)
                                                    raise HTTPException(status_code=502, detail=str(e))
                                            return handler
                                        
                                        handler = create_mcp_remote_handler(tool.name, mcp_remote)
                                        app.post(route_path, name=f"tool_{tool.name}", tags=["tools"])(handler)
                                        logger.info("Registered mcp-remote route: %s", route_path)
                                
                                await mount_mcp_remote_tool_routes(tools, mcp_remote)
                            return
                        
                        except Exception as mcp_remote_error:
                            logger.exception("mcp-remote fallback also failed: %s", mcp_remote_error)
                    else:
                        logger.warning("StdioClientTransport not available - skipping mcp-remote fallback")
                    
                    raise Exception(f"All connection methods failed: {str(e)}, {str(fe)}")

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
# --- Minimal OpenAPI augmentation for OWUI tool mapping (added) ---
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=APP_NAME,
        version=APP_VERSION,
        description=APP_DESCRIPTION,
        routes=app.routes,
    )

    # 1) Security scheme for x-api-key
    components = schema.setdefault("components", {})
    sec = components.setdefault("securitySchemes", {})
    sec["apiKeyAuth"] = {
        "type": "apiKey",
        "in": "header",
        "name": "x-api-key",
    }

    # 2) Ensure POST /tools/time is advertised and secured
    paths = schema.setdefault("paths", {})
    post_tools_time = paths.setdefault("/tools/time", {}).setdefault("post", {})
    post_tools_time.update({
        "operationId": "mcpo_time",
        "summary": "Invoke the time tool",
        "requestBody": {
            "required": False,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "echo": {"type": "string"}
                        }
                    }
                }
            }
        },
        "responses": {
            "200": {
                "description": "Successful tool response",
                "content": {
                    "application/json": {
                        "schema": {"type": "object"}
                    }
                }
            }
        },
        "security": [{"apiKeyAuth": []}],
        "tags": ["tools"],
    })

    # Optional default security
    schema["security"] = [{"apiKeyAuth": []}]

    app.openapi_schema = schema
    return app.openapi_schema

# Bind the custom OpenAPI generator
app.openapi = custom_openapi
# --- End minimal OpenAPI augmentation ---

def _collect_connector_diagnostics() -> Dict[str, Any]:
    info = {
        "mcp_version": _MCP_VERSION,
        "resolved_connector": _CONNECTOR_NAME,
        "connector_module_path": _CONNECTOR_MODULE_PATH,
        "alt_http_connect_available": bool(_ALT_HTTP_CONNECT is not None),
        "headers_configured": bool(MCP_HEADERS.strip()),
        "direct_http_fallback": "available",
        "mcp_remote_fallback": "available" if STDIO_AVAILABLE else "disabled (StdioClientTransport not found)",
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

def attach_tools_listing(app: FastAPI) -> None:
    route = f"{PATH_PREFIX.rstrip('/')}/tools" if PATH_PREFIX != "/" else "/_tools"
    @app.get(route)
    async def _tools(dep=Depends(api_dependency())):
        return {"tools": list(_DISCOVERED_TOOL_NAMES)}

attach_tools_listing(app)

def attach_tools_full_listing(app: FastAPI) -> None:
    route = f"{PATH_PREFIX.rstrip('/')}/tools_full" if PATH_PREFIX != "/" else "/_tools_full"
    @app.get(route)
    async def _tools_full(dep=Depends(api_dependency())):
        return {"tools": [dict(item) for item in _DISCOVERED_TOOLS_MIN]}

attach_tools_full_listing(app)

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
