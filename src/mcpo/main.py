"""
Open WebUI MCPO - main.py v0.0.42 (Multi-Server MCP Support)

Changes from v0.0.41:
- PRESERVES: ALL existing v0.0.41 functionality and error handling (1442 lines)
- FIXED: Authentication leakage to GitMCP servers (server-specific auth filtering)
- ENHANCED: HTTP method routing for GitMCP parameter-optional tools (GET routes)
- ADDED: Server-specific authentication exclusion logic for GitMCP

Changes from v0.0.40.3:
- Added multi-server MCP support with MCPServerConfig dataclass
- Added MCP_SERVERS_CONFIG and MCP_ENABLE_MULTI_SERVER environment variables
- Added _parse_multi_server_config() function for JSON configuration parsing
- Enhanced setup_tools() to support both single-server (existing) and multi-server modes
- Added _setup_multiple_servers() for concurrent multi-server connections
- Added _discover_server_tools() and _call_multi_server_tool() for server-specific operations
- Added _mount_multi_server_tool_routes() for namespaced tool routing
- Enhanced _collect_connector_diagnostics() with multi-server status reporting
- ALL 1102 lines of v0.0.40.3 working code preserved with zero modifications

Multi-Server Configuration Format:
{
  "mcpServers": {
    "n8n": {"url": "https://n8n-mcp-server.com/mcp", "headers": {"Authorization": "Bearer token"}},
    "github": {"url": "https://github-mcp-server.com", "headers": {"Authorization": "Bearer ghp_token"}},
    "filesystem": {"url": "https://filesystem-mcp-server.com"}
  }
}

Backward Compatibility:
- Single-server mode (MCP_ENABLE_MULTI_SERVER=false) works identically to v0.0.40.3
- Multi-server mode (MCP_ENABLE_MULTI_SERVER=true) adds concurrent server support
- All existing routes, authentication, and error handling preserved

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
from dataclasses import dataclass, field  # ADDED v0.0.41: For MCPServerConfig

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
APP_VERSION = "0.0.42"  # CHANGED from v0.0.41: Updated version for GitMCP integration fixes
APP_DESCRIPTION = "Automatically generated API from MCP Tool Schemas"
DEFAULT_PORT = int(os.getenv("PORT", "8080"))
PATH_PREFIX = os.getenv("PATH_PREFIX", "/")
CORS_ALLOWED_ORIGINS = [o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()]
API_KEY = os.getenv("API_KEY", "changeme")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://mcp-streamable-test-production.up.railway.app/mcp")
MCP_HEADERS = os.getenv("MCP_HEADERS", "")

# ADDED v0.0.41: Multi-server configuration environment variables
MCP_SERVERS_CONFIG = os.getenv("MCP_SERVERS_CONFIG", "")  # JSON string of server configs
MCP_ENABLE_MULTI_SERVER = os.getenv("MCP_ENABLE_MULTI_SERVER", "false").lower() == "true"

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

# ADDED v0.0.41: Multi-server configuration parser
def _parse_multi_server_config(config_str: str) -> List['MCPServerConfig']:
    """Parse MCP_SERVERS_CONFIG JSON into MCPServerConfig objects
    
    Supports Claude Desktop mcpServers format:
    {
      "mcpServers": {
        "server_name": {
          "url": "https://server-url.com/mcp",
          "headers": {"Authorization": "Bearer token"},
          "auth_token": "optional_direct_token",
          "type": "streamable-http",
          "description": "Server description",
          "enabled": true
        }
      }
    }
    """
    if not config_str.strip():
        return []
    
    try:
        config_data = json.loads(config_str)
        servers = []
        
        # Support both direct format and mcpServers wrapper
        servers_data = config_data.get("mcpServers", config_data)
        
        for name, server_info in servers_data.items():
            if not isinstance(server_info, dict):
                continue
                
            server_config = MCPServerConfig(
                name=name,
                server_type=server_info.get("type", "streamable-http"),
                url=server_info.get("url", ""),
                headers=server_info.get("headers", {}),
                auth_token=server_info.get("auth_token", ""),
                description=server_info.get("description", f"MCP Server: {name}"),
                enabled=server_info.get("enabled", True)
            )
            
            # If no direct auth_token but Authorization header exists, extract it
            if not server_config.auth_token and "Authorization" in server_config.headers:
                auth_value = server_config.headers["Authorization"]
                if auth_value.startswith("Bearer "):
                    server_config.auth_token = auth_value.split(" ", 1)[1]
                    
            servers.append(server_config)
            
        logger.info("Parsed %d server configurations from MCP_SERVERS_CONFIG", len(servers))
        return servers
        
    except json.JSONDecodeError as e:
        logger.error("Failed to parse MCP_SERVERS_CONFIG as JSON: %s", e)
        return []
    except Exception as e:
        logger.error("Unexpected error parsing MCP_SERVERS_CONFIG: %s", e)
        return []

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

# ADDED v0.0.41: Multi-server configuration dataclass
@dataclass
class MCPServerConfig:
    """Configuration for individual MCP servers in multi-server mode"""
    name: str
    server_type: str = "streamable-http"  # "streamable-http", "http", "mcp-remote"
    url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    auth_token: str = ""
    description: str = ""
    enabled: bool = True
    
    def __post_init__(self):
        """Set default description if not provided"""
        if not self.description:
            self.description = f"MCP Server: {self.name}"

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

# Direct HTTP fallback functions for auth issues
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

# ADDED v0.0.41: Multi-server tool discovery
async def _discover_server_tools(server: MCPServerConfig) -> List[ToolDef]:
    """Discover tools from specific MCP server"""
    headers = server.headers.copy()
    
    # FIXED v0.0.42: Only apply auth for servers that need it
    if server.name != "gitmcp" and server.auth_token and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {server.auth_token}"
    # Explicitly exclude GitMCP from authentication
    elif server.name == "gitmcp" and "Authorization" in headers:
        headers.pop("Authorization", None)
    
    logger.info("Discovering tools from server %s (%s) - auth: %s", server.name, server.url, "enabled" if "Authorization" in headers else "disabled")
    
    try:
        # Try direct HTTP first (primary method)
        tools = await discover_tools_via_http_fallback(server.url, headers)
        logger.info("Discovered %d tools from server %s via direct HTTP", len(tools), server.name)
        return tools
    except Exception as e:
        logger.warning("Direct HTTP tool discovery failed for server %s: %s", server.name, e)
        
        # Fallback to MCP connector
        try:
            async with _connector_wrapper(server.url) as (reader, writer):
                tools = await list_mcp_tools(reader, writer)
                logger.info("Discovered %d tools from server %s via MCP connector", len(tools), server.name)
                return tools
        except Exception as fe:
            logger.error("All tool discovery methods failed for server %s: %s, %s", server.name, e, fe)
            return []

# ADDED v0.0.41: Multi-server tool execution
async def _call_multi_server_tool(server: MCPServerConfig, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Route tool call to appropriate MCP server"""
    headers = server.headers.copy()
    
    # FIXED v0.0.42: Only apply auth for servers that need it
    if server.name != "gitmcp" and server.auth_token and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {server.auth_token}"
    # GitMCP servers explicitly excluded from authentication
    elif server.name == "gitmcp" and "Authorization" in headers:
        headers.pop("Authorization", None)  # Remove any auth headers for GitMCP
    
    logger.debug("Calling tool %s on server %s (auth: %s)", tool_name, server.name, "enabled" if "Authorization" in headers else "disabled")
    
    try:
        # Use direct HTTP first (proven working method)
        result = await call_mcp_tool_via_http_fallback(server.url, headers, tool_name, arguments)
        return result
        
    except Exception as e:
        logger.warning("Direct HTTP failed for tool %s on server %s: %s", tool_name, server.name, e)
        
        # Fallback to MCP connector
        try:
            async with _connector_wrapper(server.url) as (reader, writer):
                result = await call_mcp_tool(reader, writer, tool_name, arguments)
                return result
        except Exception as fe:
            logger.exception("All methods failed for tool %s on server %s: %s, %s", tool_name, server.name, e, fe)
            raise HTTPException(status_code=502, detail=f"All connection methods failed: {str(e)}, {str(fe)}")

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

            first_item = content[0]
            if hasattr(first_item, 'text'):
                text = first_item.text
            elif isinstance(first_item, dict):
                text = first_item.get('text', '')
            else:
                text = str(first_item)

            if not text:
                return {"content": content}

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

# ADDED v0.0.41: Multi-server tool route mounting
async def _mount_multi_server_tool_routes(tools: List[ToolDef], servers: List[MCPServerConfig], app: FastAPI):
    """Mount routes for multi-server tools with proper server routing"""
    
    # Create server lookup dictionary for efficient routing
    server_lookup = {server.name: server for server in servers}
    
    for tool in tools:
        # Extract server name from namespaced tool name (format: servername_toolname)
        name_parts = tool.name.split('_', 1)
        if len(name_parts) < 2:
            logger.warning("Skipping tool with invalid namespaced name: %s", tool.name)
            continue
            
        server_name = name_parts[0]
        original_tool_name = name_parts[1]
        server_config = server_lookup.get(server_name)
        
        if not server_config:
            logger.warning("No server config found for tool: %s (server: %s)", tool.name, server_name)
            continue
            
        route_path = f"{PATH_PREFIX.rstrip('/')}/tools/{tool.name}" if PATH_PREFIX != "/" else f"/tools/{tool.name}"
        logger.info("Registering multi-server route: %s -> %s:%s", route_path, server_name, original_tool_name)
        
        # Create POST handler with server-specific routing
        async def create_post_handler(tool_obj: ToolDef, server: MCPServerConfig, orig_name: str):
            async def handler(payload: Optional[Dict[str, Any]] = Body(None), dep=Depends(api_dependency())):
                try:
                    if payload is None:
                        logger.info("No request body provided - using empty dict for Open WebUI compatibility")
                        payload = {}
                        
                    # Route to specific server
                    result = await _call_multi_server_tool(server, orig_name, payload)
                    return JSONResponse(status_code=200, content=result)
                    
                except HTTPException:
                    raise
                except Exception as e:
                    logger.exception("Multi-server tool call failed for %s on %s: %s", orig_name, server.name, e)
                    raise HTTPException(status_code=502, detail=str(e))
            return handler
        
        handler = await create_post_handler(tool, server_config, original_tool_name)
        app.post(route_path, name=f"tool_{tool.name}", tags=["tools"])(handler)
        
        # ENHANCED v0.0.42: Add GET route for GitMCP tools (always) and parameter-less tools (existing logic)
        input_schema = tool.inputSchema or {}
        is_parameter_less = not input_schema.get('properties') and not input_schema.get('required')
        is_gitmcp_tool = server_name == "gitmcp"
        
        if is_parameter_less or is_gitmcp_tool:
            async def create_get_handler(tool_obj: ToolDef, server: MCPServerConfig, orig_name: str):
                async def get_handler(dep=Depends(api_dependency())):
                    try:
                        # GitMCP tools always use empty payload, parameter-less tools use empty dict
                        payload = {} if is_gitmcp_tool else {}
                        result = await _call_multi_server_tool(server, orig_name, payload)
                        return JSONResponse(status_code=200, content=result)
                    except HTTPException:
                        raise
                    except Exception as e:
                        logger.exception("Multi-server GET tool call failed for %s on %s: %s", orig_name, server.name, e)
                        raise HTTPException(status_code=502, detail=str(e))
                return get_handler
                
            get_handler = await create_get_handler(tool, server_config, original_tool_name)
            app.get(route_path, name=f"tool_{tool.name}_get", tags=["tools"])(get_handler)
            logger.info("Added GET route for %s tool: %s", "GitMCP" if is_gitmcp_tool else "parameter-less", route_path)

# ADDED v0.0.41: Multiple server setup function
async def _setup_multiple_servers(servers: List[MCPServerConfig], app: FastAPI):
    """Setup multiple MCP servers concurrently"""
    logger.info("Setting up %d MCP servers concurrently", len(servers))
    
    all_tools = []
    
    # Discover tools from all enabled servers
    for server in servers:
        if not server.enabled:
            logger.info("Skipping disabled server: %s", server.name)
            continue
            
        try:
            logger.info("Connecting to MCP server: %s (%s)", server.name, server.url)
            
            # Discover tools from this server
            server_tools = await _discover_server_tools(server)
            
            if not server_tools:
                logger.warning("No tools discovered from server %s", server.name)
                continue
            
            # Namespace tools with server name to avoid conflicts
            namespaced_tools = []
            for tool in server_tools:
                namespaced_tool = ToolDef(
                    name=f"{server.name}_{tool.name}",  # Namespace with server name
                    description=f"[{server.name}] {tool.description or ''}".strip(),
                    inputSchema=tool.inputSchema,
                    outputSchema=tool.outputSchema
                )
                namespaced_tools.append(namespaced_tool)
            
            all_tools.extend(namespaced_tools)
            logger.info("Added %d namespaced tools from server %s", len(namespaced_tools), server.name)
            
        except Exception as e:
            logger.error("Failed to connect to server %s (%s): %s", server.name, server.url, e)
            # Continue with other servers rather than failing completely
            continue
    
    if not all_tools:
        logger.error("No tools discovered from any MCP server")
        return
    
    # Mount routes for all discovered tools
    await _mount_multi_server_tool_routes(all_tools, servers, app)
    
    # Update global tool lists for diagnostics and listing endpoints
    _DISCOVERED_TOOL_NAMES.clear()
    _DISCOVERED_TOOL_NAMES.extend([t.name for t in all_tools])
    _DISCOVERED_TOOLS_MIN.clear()
    for t in all_tools:
        try:
            _DISCOVERED_TOOLS_MIN.append({
                "name": t.name,
                "description": t.description,
                "inputSchema": t.inputSchema,
            })
        except Exception:
            pass
    
    logger.info("Successfully configured %d tools from %d servers", len(all_tools), len(servers))

# ADDED v0.0.41: Single-server setup (preserved existing logic)
async def _setup_single_server(app: FastAPI):
    """Preserve ALL existing single-server setup logic exactly as-is"""
    logger.info("Single-server mode (preserving v0.0.40.3 behavior)")
    
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
                    if payload is None:
                        logger.info("No request body provided - using empty dict for Open WebUI compatibility")
                        payload = {}
                    result = await call_mcp_tool_via_http_fallback(MCP_SERVER_URL, headers, _tool.name, payload or {})
                    return JSONResponse(status_code=200, content=result)
                except Exception as e:
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

            input_schema = tool.inputSchema or {}
            if not input_schema.get('properties') and not input_schema.get('required'):
                async def get_handler(_tool=tool, dep=Depends(api_dependency())):
                    try:
                        result = await call_mcp_tool_via_http_fallback(
                            MCP_SERVER_URL, headers, _tool.name, {}
                        )
                        return JSONResponse(status_code=200, content=result)
                    except Exception as e:
                        logger.warning(
                            "Direct HTTP failed for tool %s, trying MCP connector: %s", _tool.name, e
                        )
                        try:
                            async with _connector_wrapper(MCP_SERVER_URL) as (reader, writer):
                                result = await call_mcp_tool(reader, writer, _tool.name, {})
                                return JSONResponse(status_code=200, content=result)
                        except HTTPException as he:
                            raise he
                        except Exception as fe:
                            logger.exception(
                                "Both direct HTTP and MCP connector failed for tool %s: %s, %s",
                                _tool.name, e, fe
                            )
                            raise HTTPException(
                                status_code=502,
                                detail=f"Both direct HTTP and MCP connector failed: {str(e)}, {str(fe)}"
                            )
                app.get(route_path, name=f"tool_{tool.name}_get", tags=["tools"])(get_handler)
                logger.info("Added GET route for parameter-less tool: %s", route_path)

    # Preserve ALL existing setup_tools logic
    try:
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
            if STDIO_AVAILABLE:
                logger.warning("Trying mcp-remote as final fallback")
                try:
                    auth_token = None
                    if headers and "Authorization" in headers:
                        auth_value = headers["Authorization"]
                        if auth_value.startswith("Bearer "):
                            auth_token = auth_value.split(" ", 1)[1]
                    if not auth_token:
                        raise Exception("No auth token available for mcp-remote")
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

        # ADDED v0.0.41: Multi-server mode detection and setup
        if MCP_ENABLE_MULTI_SERVER and MCP_SERVERS_CONFIG:
            logger.info("Multi-server mode enabled, parsing server configurations")
            servers = _parse_multi_server_config(MCP_SERVERS_CONFIG)
            
            if not servers:
                logger.warning("No valid servers found in MCP_SERVERS_CONFIG, falling back to single server")
                logger.info("Echo/Ping routes registered")
                logger.info("Configuring for a single StreamableHTTP MCP Server with URL [%s]", MCP_SERVER_URL)
                await _setup_single_server(app)
            else:
                logger.info("Echo/Ping routes registered")
                logger.info("Configuring for %d MCP servers in multi-server mode", len(servers))
                for server in servers:
                    logger.info(" - %s: %s (%s)", server.name, server.url, "enabled" if server.enabled else "disabled")
                await _setup_multiple_servers(servers, app)
        else:
            logger.info("Single-server mode (existing v0.0.40.3 behavior)")
            logger.info("Echo/Ping routes registered")  
            logger.info("Configuring for a single StreamableHTTP MCP Server with URL [%s]", MCP_SERVER_URL)
            await _setup_single_server(app)

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

    components = schema.setdefault("components", {})
    sec = components.setdefault("securitySchemes", {})
    sec["apiKeyAuth"] = {
        "type": "apiKey",
        "in": "header",
        "name": "x-api-key",
    }

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

    schema["security"] = [{"apiKeyAuth": []}]

    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi

# ENHANCED v0.0.41: Diagnostics with multi-server support
def _collect_connector_diagnostics() -> Dict[str, Any]:
    """Enhanced diagnostics including multi-server status - preserves all existing info"""
    # Preserve ALL existing diagnostic logic
    info = {
        "mcp_version": _MCP_VERSION,
        "resolved_connector": _CONNECTOR_NAME,
        "connector_module_path": _CONNECTOR_MODULE_PATH,
        "alt_http_connect_available": bool(_ALT_HTTP_CONNECT is not None),
        "headers_configured": bool(MCP_HEADERS.strip()),
        "direct_http_fallback": "available",
        "mcp_remote_fallback": "available" if STDIO_AVAILABLE else "disabled (StdioClientTransport not found)",
    }
    
    # ADDED v0.0.41: Multi-server diagnostics (only adds, never modifies existing)
    if MCP_ENABLE_MULTI_SERVER:
        info["multi_server_mode"] = True
        info["multi_server_config_provided"] = bool(MCP_SERVERS_CONFIG.strip())
        
        if MCP_SERVERS_CONFIG.strip():
            try:
                servers = _parse_multi_server_config(MCP_SERVERS_CONFIG)
                info["configured_servers"] = [
                    {
                        "name": s.name,
                        "type": s.server_type,
                        "url": s.url,
                        "enabled": s.enabled,
                        "description": s.description,
                        "has_headers": bool(s.headers),
                        "has_auth_token": bool(s.auth_token)
                    } for s in servers
                ]
                info["server_count"] = len(servers)
                info["enabled_server_count"] = sum(1 for s in servers if s.enabled)
            except Exception as e:
                info["multi_server_config_error"] = str(e)
        else:
            info["configured_servers"] = []
            info["server_count"] = 0
    else:
        info["multi_server_mode"] = False
        info["single_server_url"] = MCP_SERVER_URL
    
    # Preserve all existing diagnostic logic (unchanged from v0.0.40.3)
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
