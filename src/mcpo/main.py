"""
Open WebUI MCPO - main.py v0.0.68 (Add detailed logging to debug exactly where the parsing fails)

Changes from v0.0.65:
- FIXED: Context7 tool invocation failure by tracking successful connection methods during discovery
- ADDED: SERVER_CONNECTION_METHODS global dict to track which connection method worked per server
- ENHANCED: _discover_server_tools() now stores successful connection method for each server
- IMPROVED: _call_multi_server_tool() uses the same connection method that worked during discovery
- PRESERVED: ALL existing functionality for n8n-mcp and other working servers

Tool Invocation Fix Applied:
1. Added SERVER_CONNECTION_METHODS = {} to track connection method success per server
2. Enhanced _discover_server_tools() to store successful method ("direct_http" or "mcp_connector")
3. Modified _call_multi_server_tool() to check stored method and route accordingly
4. Context7 servers that discovered tools via MCP connector now skip direct HTTP for tool calls
5. Maintains full backward compatibility and existing fallback logic

Changes from v0.0.61:
- GitMCP integration temporarily disabled due to protocol compatibility issues (mcp-remote 0.1.29 hardcoded protocol version bug)
- ALL n8n-mcp functionality preserved (39 tools registered and working)
- Added Context7 preparation comments for future Open WebUI Pipelines integration
- Maintained backward compatibility with existing configurations
- Prepared monitoring capability for GitMCP protocol fixes

GitMCP Status v0.0.62:
- GitMCP detection and routing COMMENTED OUT (lines 385, 610, 690, 1190)
- GitMCP servers will be skipped during multi-server initialization
- n8n-mcp server continues to work normally via direct HTTP and MCP protocols
- Context7 integration to be implemented via Open WebUI Pipelines (separate from this proxy)

Preserved from v0.0.61:
- ALL existing v0.0.61 functionality for n8n-mcp server (1572 lines preserved)
- Multi-server configuration support (for future GitMCP restoration)
- Request body tolerance and response formatting improvements
- Error handling and authentication logic
- Direct HTTP fallback and MCP connector methods

Changes from v0.0.60:
- PRESERVES: ALL existing v0.0.60 GitMCP detection and v0.0.45 request body tolerance and v0.0.44 response formatting (1572 lines)
- FIXED: GitMCP connection method - replaced direct MCP client with MCPRemoteManager (npx mcp-remote wrapper)
- ENHANCED: GitMCP servers now use proper npx mcp-remote subprocess approach (like Claude Desktop)
- ADDED: GitMCP routing through existing MCPRemoteManager class for proper NPX wrapper integration
- IMPROVED: GitMCP tool discovery and execution via npx subprocess instead of direct MCP protocol

GitMCP NPX Wrapper Integration Fixes Applied:
1. Fixed GitMCP connections to use MCPRemoteManager (npx mcp-remote) instead of direct MCP client
2. GitMCP servers now use npx subprocess wrapper approach (like Claude Desktop)
3. Replaced direct _connector_wrapper() for GitMCP with MCPRemoteManager routing
4. Eliminated "TaskGroup sub-exception" errors by using proper NPX wrapper approach
5. All v0.0.46 GitMCP detection, v0.0.45 request body tolerance, and v0.0.44 response formatting preserved

Changes from v0.0.45:
- PRESERVES: ALL existing v0.0.45 functionality and response formatting (1556 lines)
- FIXED: GitMCP protocol integration - replaced HTTP calls with proper MCP client protocol
- ENHANCED: GitMCP servers now use MCP client connections instead of HTTP fallback
- ADDED: Dedicated GitMCP MCP protocol handling for proper WebSocket/SSE communication
- IMPROVED: GitMCP tool discovery and execution via native MCP protocol (like Claude Desktop)

GitMCP Protocol Integration Fixes Applied:
1. Fixed GitMCP connections to use proper MCP client protocol instead of HTTP calls
2. GitMCP servers now use MCP WebSocket/SSE protocol communication (like Claude Desktop)
3. Replaced HTTP fallback for GitMCP with proper MCP client protocol
4. Maintained HTTP API compatibility for Open WebUI while using MCP backend for GitMCP
5. All v0.0.45 request body tolerance and v0.0.44 response formatting preserved

Changes from v0.0.44:
- PRESERVES: ALL existing v0.0.44 functionality and response formatting (1515 lines)
- FIXED: MCP-remote handler empty body rejection (line 1312)
- ENHANCED: Request body tolerance for Open WebUI lightweight requests
- ADDED: Better default parameter handling for empty request bodies
- IMPROVED: Body parsing flexibility for minimal Open WebUI requests

Request Body Tolerance Fixes Applied:
1. Fixed mcp-remote handler to accept Optional[Dict] with Body(None)
2. Enhanced empty body handling with better default parameter generation
3. Improved request payload tolerance for minimal Open WebUI patterns
4. All v0.0.44 response formatting improvements preserved

Changes from v0.0.43:
- PRESERVES: ALL existing v0.0.43 functionality and error handling (1514 lines)
- FIXED: Response format consistency for Open WebUI (lines 500-514, 694-717, 943-953)
- FIXED: Empty tool response handling to always return structured content (lines 502-503, 701-702, 944-945)
- FIXED: Error response structure consistency (lines 496-497, 921-927, 936-940)
- ENHANCED: Content array preservation for Open WebUI expectations

Protocol Fixes Applied:
1. Standardized response format: Always returns {"result": content} for Open WebUI compatibility
2. Empty responses return {"result": ""} instead of {} for consistency
3. Error responses unified under {"error": message} structure
4. Content arrays preserved without premature JSON parsing

Changes from v0.0.42:
- PRESERVES: ALL existing v0.0.42 functionality and error handling (1317 lines)
- FIXED: Open WebUI request format handling for GitMCP tools (enhanced POST body processing)
- ENHANCED: Request body validation and fallback logic for malformed Open WebUI requests
- ADDED: Content-Type handling and robust payload normalization for GitMCP compatibility

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
    "gitmcp": {"url": "https://gitmcp.io/DC-AAIA/n8n-mcp"}
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

from fastapi import FastAPI, Depends, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_core import ValidationError as PydValidationError
from starlette.responses import JSONResponse

from mcp.client.session import ClientSession
from mcp.shared.exceptions import McpError

# Try to import stdio transport - may not be available in all MCP versions
try:
    from mcp.client.stdio import StdioClientTransport
    STDIOAVAILABLE = True
except ImportError:
    try:
        # Try alternative import paths for different MCP versions
        from mcp.client.stdio import stdio_client as StdioClientTransport
        STDIOAVAILABLE = True
    except ImportError:
        try:
            from mcp.client import StdioClientTransport
            STDIOAVAILABLE = True
        except ImportError:
            try:
                from mcp import StdioClientTransport
                STDIOAVAILABLE = True
            except ImportError:
                STDIOAVAILABLE = False
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
APP_VERSION = "0.0.68"  # CHANGED from v0.0.67: Add detailed logging to debug exactly where the parsing fails
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
# ADDED v0.0.66: Track successful connection methods per server for tool invocation
SERVER_CONNECTION_METHODS = {}  # Dict[server_name, connection_method] - "direct_http" or "mcp_connector"

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

# ADDED v0.0.43: Enhanced request body processing for Open WebUI compatibility
async def _process_open_webui_payload(request: Request, raw_payload: Any) -> Dict[str, Any]:
    """Process and normalize Open WebUI request payloads for GitMCP compatibility

    Handles various malformed request scenarios from Open WebUI:
    - Null/undefined bodies
    - Empty strings
    - Malformed JSON
    - Missing Content-Type headers
    """
    try:
        # Check Content-Type for additional context
        content_type = request.headers.get("content-type", "").lower()
        logger.debug("Processing Open WebUI payload - Content-Type: %s, Raw payload type: %s",
                    content_type, type(raw_payload).__name__)

        # Handle null/None payloads (most common Open WebUI issue)
        if raw_payload is None:
            logger.info("Null payload detected - normalizing to empty dict for GitMCP compatibility")
            return {}

        # Handle empty string payloads
        if isinstance(raw_payload, str):
            if not raw_payload.strip():
                logger.info("Empty string payload detected - normalizing to empty dict")
                return {}
            # Try to parse as JSON
            try:
                parsed = json.loads(raw_payload)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                logger.warning("Malformed JSON string payload: %s", raw_payload[:100])
                return {}

        # Handle dictionary payloads (normal case)
        if isinstance(raw_payload, dict):
            return raw_payload

        # Handle list payloads (convert to dict with data key)
        if isinstance(raw_payload, list):
            logger.info("List payload detected - wrapping in dict for GitMCP compatibility")
            return {"data": raw_payload}

        # Handle other types
        logger.info("Unexpected payload type %s - normalizing to empty dict", type(raw_payload).__name__)
        return {}

    except Exception as e:
        logger.error("Payload processing error: %s - falling back to empty dict", e)
        return {}

# ADDED v0.0.44: Standardized response formatting for Open WebUI
def _format_tool_response(content: Any) -> Dict[str, Any]:
    """Format tool response consistently for Open WebUI compatibility

    Protocol Fix: Ensures all responses follow {"result": content} structure
    """
    # If already has result key, return as-is
    if isinstance(content, dict) and "result" in content:
        return content

    # If has content key, wrap in result
    if isinstance(content, dict) and "content" in content:
        return {"result": content["content"]}

    # If has raw key, preserve text content
    if isinstance(content, dict) and "raw" in content:
        return {"result": content["raw"]}

    # Empty responses return empty string for consistency
    if not content:
        return {"result": ""}

    # Wrap any other content in result
    return {"result": content}

# ADDED v0.0.45: Enhanced request body tolerance for Open WebUI
def _ensure_request_compatibility(payload: Any, tool_schema: Dict[str, Any] = None) -> Dict[str, Any]:
    """Ensure request payload compatibility with Open WebUI lightweight requests

    Handles Open WebUI's minimal request patterns:
    - Empty POST bodies
    - Minimal request payloads
    - Missing required parameters with sensible defaults
    """
    # Handle None/empty payloads
    if payload is None:
        payload = {}

    # Convert non-dict payloads to dict
    if not isinstance(payload, dict):
        payload = {}

    # If tool expects parameters but payload is empty, provide defaults based on schema
    if tool_schema and not payload:
        properties = tool_schema.get('properties', {})
        required = tool_schema.get('required', [])

        # Generate minimal defaults for required parameters
        for param in required:
            if param not in payload:
                param_schema = properties.get(param, {})
                param_type = param_schema.get('type', 'string')

                # Provide sensible defaults based on type
                if param_type == 'string':
                    payload[param] = param_schema.get('default', '')
                elif param_type == 'number' or param_type == 'integer':
                    payload[param] = param_schema.get('default', 0)
                elif param_type == 'boolean':
                    payload[param] = param_schema.get('default', False)
                elif param_type == 'array':
                    payload[param] = param_schema.get('default', [])
                elif param_type == 'object':
                    payload[param] = param_schema.get('default', {})
                else:
                    payload[param] = param_schema.get('default', '')

    return payload


# ADDED v0.0.65: Custom request dependency to handle empty JSON bodies
async def parse_optional_json_body(request: Request) -> Dict[str, Any]:
    """Parse request body with proper empty JSON handling for Open WebUI compatibility"""
    try:
        body = await request.body()
        if not body:
            logger.debug("No request body - returning empty dict")
            return {}
            
        content_type = request.headers.get("content-type", "").lower()
        if "application/json" not in content_type:
            logger.debug("Content-Type not JSON: %s - returning empty dict", content_type)
            return {}
            
        body_str = body.decode('utf-8')
        logger.debug("Raw body string: %s", body_str[:200])  # Log first 200 chars
        
        if not body_str.strip():
            logger.debug("Empty body string - returning empty dict")
            return {}
            
        # This is the key fix - handle empty objects properly
        if body_str.strip() == '{}':
            logger.debug("Empty JSON object - returning empty dict")
            return {}
            
        parsed = json.loads(body_str)
        logger.debug("Successfully parsed JSON: %s", parsed)
        return parsed
        
    except json.JSONDecodeError as jde:
        logger.error("JSON decode error: %s - body was: %s", jde, body_str[:200])
        return {}
    except Exception as e:
        logger.error("Request body parsing failed: %s - returning empty dict", e)
        return {}

# DISABLED v0.0.62: GitMCP-specific MCP protocol detection (commented out)
# def _is_gitmcp_server(server_name: str, server_url: str) -> bool:
#     """Detect if server is GitMCP requiring proper MCP protocol instead of HTTP"""
#     # Check by server name
#     if server_name and server_name.lower() == "gitmcp":
#         return True
# 
#     # Check by URL pattern
#     if server_url and "gitmcp.io" in server_url.lower():
#         return True
# 
#     return False

# ADDED v0.0.62: GitMCP detection function that always returns False (disabled)
def _is_gitmcp_server(server_name: str, server_url: str) -> bool:
    """GitMCP detection disabled in v0.0.62 - always returns False
    
    GitMCP integration temporarily disabled due to protocol compatibility issues.
    See Context7 integration via Open WebUI Pipelines as alternative.
    """
    # Log GitMCP detection attempts for monitoring
    if server_name and server_name.lower() == "gitmcp":
        logger.info("GitMCP server detected (%s) but integration disabled in v0.0.62", server_name)
    elif server_url and "gitmcp.io" in server_url.lower():
        logger.info("GitMCP URL detected (%s) but integration disabled in v0.0.62", server_url)
    
    return False  # Always return False to disable GitMCP routing

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

        # FIXED v0.0.44: Unified error response structure
        if "error" in result:
            raise HTTPException(status_code=400, detail={"error": result["error"]})

        tool_result = result.get("result", {})
        content = tool_result.get("content", [])

        # FIXED v0.0.44: Consistent response formatting for Open WebUI
        if not content:
            return {"result": ""}

        # Preserve full content array for Open WebUI
        first = content[0] if content else {}
        text = first.get("text") if isinstance(first, dict) else None

        if text:
            # Try to parse as JSON but preserve structure
            try:
                parsed = json.loads(text)
                return {"result": parsed}
            except Exception:
                return {"result": text}

        # Return full content array wrapped in result
        return {"result": content}

# ADDED v0.0.41: Multi-server tool discovery
async def _discover_server_tools(server: MCPServerConfig) -> List[ToolDef]:
    """Discover tools from specific MCP server"""
    headers = server.headers.copy()

    # DISABLED v0.0.62: GitMCP-specific authentication logic (commented out but preserved)
    # FIXED v0.0.42: Only apply auth for servers that need it
    # if server.name != "gitmcp" and server.auth_token and "Authorization" not in headers:
    #     headers["Authorization"] = f"Bearer {server.auth_token}"
    # # Explicitly exclude GitMCP from authentication
    # elif server.name == "gitmcp" and "Authorization" in headers:
    #     headers.pop("Authorization", None)

    # ADDED v0.0.62: Apply authentication for all non-GitMCP servers
    if server.auth_token and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {server.auth_token}"

    logger.info("Discovering tools from server %s (%s) - auth: %s", server.name, server.url, "enabled" if "Authorization" in headers else "disabled")

    # DISABLED v0.0.62: GitMCP routing (commented out but preserved for future restoration)
    # ADDED v0.0.47: Use MCPRemoteManager (npx mcp-remote) for GitMCP servers
    # if _is_gitmcp_server(server.name, server.url):
    #     logger.info("Detected GitMCP server %s - using npx mcp-remote wrapper instead of direct MCP client", server.name)
    #     try:
    #         # Use MCPRemoteManager for GitMCP (npx mcp-remote subprocess like Claude Desktop)
    #         mcp_remote = MCPRemoteManager()
    #         # GitMCP doesn't require auth token for npx mcp-remote
    #         await mcp_remote.start(server.url, "dummy_token")  # MCPRemoteManager expects auth_token but GitMCP ignores it
    #         try:
    #             tools = await mcp_remote.list_tools()
    #             logger.info("Discovered %d tools from GitMCP server %s via npx mcp-remote", len(tools), server.name)
    #             return tools
    #         finally:
    #             await mcp_remote.stop()
    #     except Exception as e:
    #         logger.error("GitMCP npx mcp-remote connection failed for server %s: %s", server.name, e)
    #         return []

    # ADDED v0.0.62: Skip GitMCP servers entirely
    if _is_gitmcp_server(server.name, server.url):
        logger.info("GitMCP server %s skipped - integration disabled in v0.0.62", server.name)
        return []

    # Try direct HTTP first (primary method for non-GitMCP servers)
    try:
        tools = await discover_tools_via_http_fallback(server.url, headers)
        logger.info("Discovered %d tools from server %s via direct HTTP", len(tools), server.name)
        
        # ADDED v0.0.66: Track successful connection method
        SERVER_CONNECTION_METHODS[server.name] = "direct_http"
        logger.debug("Stored connection method for %s: direct_http", server.name)
        
        return tools
    except Exception as e:
        logger.warning("Direct HTTP tool discovery failed for server %s: %s", server.name, e)

        # Fallback to MCP connector
        try:
            async with _connector_wrapper(server.url) as (reader, writer):
                tools = await list_mcp_tools(reader, writer)
                logger.info("Discovered %d tools from server %s via MCP connector", len(tools), server.name)
                
                # ADDED v0.0.66: Track successful connection method
                SERVER_CONNECTION_METHODS[server.name] = "mcp_connector"
                logger.debug("Stored connection method for %s: mcp_connector", server.name)
                
                return tools
        except Exception as fe:
            logger.error("All tool discovery methods failed for server %s: %s, %s", server.name, e, fe)
            
            # ADDED v0.0.66: Don't store connection method if both failed
            SERVER_CONNECTION_METHODS.pop(server.name, None)
            
            return []

# ADDED v0.0.41: Multi-server tool execution
async def _call_multi_server_tool(server: MCPServerConfig, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Route tool call to appropriate MCP server"""
    headers = server.headers.copy()

    # DISABLED v0.0.62: GitMCP-specific authentication logic (commented out but preserved)
    # FIXED v0.0.42: Only apply auth for servers that need it
    # if server.name != "gitmcp" and server.auth_token and "Authorization" not in headers:
    #     headers["Authorization"] = f"Bearer {server.auth_token}"
    # # GitMCP servers explicitly excluded from authentication
    # elif server.name == "gitmcp" and "Authorization" in headers:
    #     headers.pop("Authorization", None)  # Remove any auth headers for GitMCP

    # ADDED v0.0.62: Apply authentication for all non-GitMCP servers
    if server.auth_token and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {server.auth_token}"

    logger.debug("Calling tool %s on server %s (auth: %s)", tool_name, server.name, "enabled" if "Authorization" in headers else "disabled")
    
    # ADDED v0.0.66: Use the connection method that worked during discovery
    stored_method = SERVER_CONNECTION_METHODS.get(server.name)
    logger.debug("Server %s stored connection method: %s", server.name, stored_method)

    if stored_method == "mcp_connector":
        # Use MCP connector directly since it worked during discovery
        logger.debug("Using MCP connector for %s (stored method)", server.name)
        try:
            async with _connector_wrapper(server.url) as (reader, writer):
                result = await call_mcp_tool(reader, writer, tool_name, arguments)
                # FIXED v0.0.44: Apply consistent response formatting
                return _format_tool_response(result)
        except Exception as e:
            logger.exception("MCP connector tool call failed for %s on %s: %s", tool_name, server.name, e)
            raise HTTPException(status_code=502, detail=f"MCP connector failed: {str(e)}")
    
    elif stored_method == "direct_http":
        # Use direct HTTP since it worked during discovery
        logger.debug("Using direct HTTP for %s (stored method)", server.name)
        try:
            result = await call_mcp_tool_via_http_fallback(server.url, headers, tool_name, arguments)
            # FIXED v0.0.44: Apply consistent response formatting
            return _format_tool_response(result)
        except Exception as e:
            logger.exception("Direct HTTP tool call failed for %s on %s: %s", tool_name, server.name, e)
            raise HTTPException(status_code=502, detail=f"Direct HTTP failed: {str(e)}")
    
    # No stored method or unknown method - fall back to original logic
    logger.warning("No stored connection method for server %s, using fallback logic", server.name)

    # DISABLED v0.0.62: GitMCP routing (commented out but preserved for future restoration)
    # ADDED v0.0.47: Use MCPRemoteManager (npx mcp-remote) for GitMCP servers
    # if _is_gitmcp_server(server.name, server.url):
    #     logger.debug("Using npx mcp-remote for GitMCP server %s tool %s", server.name, tool_name)
    #     try:
    #         # Use MCPRemoteManager for GitMCP (npx mcp-remote subprocess like Claude Desktop)
    #         mcp_remote = MCPRemoteManager()
    #         # GitMCP doesn't require auth token for npx mcp-remote
    #         await mcp_remote.start(server.url, "dummy_token")  # MCPRemoteManager expects auth_token but GitMCP ignores it
    #         try:
    #             result = await mcp_remote.call_tool(tool_name, arguments)
    #             # FIXED v0.0.44: Apply consistent response formatting
    #             return _format_tool_response(result)
    #         finally:
    #             await mcp_remote.stop()
    #     except Exception as e:
    #         logger.exception("GitMCP npx mcp-remote tool call failed for %s on %s: %s", tool_name, server.name, e)
    #         raise HTTPException(status_code=502, detail=f"GitMCP npx mcp-remote connection failed: {str(e)}")

    # ADDED v0.0.62: Reject GitMCP tool calls with clear error message
    if _is_gitmcp_server(server.name, server.url):
        logger.warning("Tool call to GitMCP server %s rejected - integration disabled in v0.0.62", server.name)
        raise HTTPException(status_code=503, detail=f"GitMCP integration disabled - use Context7 alternative for documentation access")

    try:
        # Use direct HTTP first (proven working method for non-GitMCP servers)
        result = await call_mcp_tool_via_http_fallback(server.url, headers, tool_name, arguments)
        # FIXED v0.0.44: Apply consistent response formatting
        return _format_tool_response(result)

    except Exception as e:
        logger.warning("Direct HTTP failed for tool %s on server %s: %s", tool_name, server.name, e)

        # Fallback to MCP connector
        try:
            async with _connector_wrapper(server.url) as (reader, writer):
                result = await call_mcp_tool(reader, writer, tool_name, arguments)
                # FIXED v0.0.44: Apply consistent response formatting
                return _format_tool_response(result)
        except Exception as fe:
            logger.exception("All methods failed for tool %s on server %s: %s, %s", tool_name, server.name, e, fe)
            raise HTTPException(status_code=502, detail=f"All connection methods failed: {str(e)}, {str(fe)}")

class MCPRemoteManager:
    """Manages mcp-remote subprocess for HTTP authentication bridge"""
    def __init__(self):
        self.process = None
        self.transport = None
        self.session = None

    async def start(self, url: str, authtoken: str):
        if not STDIOAVAILABLE:
            raise RuntimeError("StdioClientTransport not available - cannot use mcp-remote fallback")
        
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        
        # DISABLED v0.0.62: GitMCP-specific logic (commented out but preserved)
        # GitMCP uses standard mcp-remote without any flags (per official documentation)
        # if "gitmcp.io" in url.lower():
        #     # GitMCP is a public service - no authentication or protocol flags needed
        #     cmd_args = ["npx", "-y", "mcp-remote", url]
        #     self.protocol_version = "auto-negotiated"
        #     self.client_name = "mcpo-gitmcp-client"
        # else:
        #     # Standard MCP servers (like n8n) use modern protocol
        #     if authtoken and authtoken != "dummytoken":
        #         cmd_args = ["npx", "-y", "mcp-remote", url, "--header", f"Authorization: Bearer {authtoken}"]
        #     else:
        #         cmd_args = ["npx", "-y", "mcp-remote", url]
        #     self.protocol_version = "2025-06-18"
        #     self.client_name = "mcpo-client"

        # ADDED v0.0.62: Standard MCP servers only (GitMCP disabled)
        if authtoken and authtoken != "dummytoken":
            cmd_args = ["npx", "-y", "mcp-remote", url, "--header", f"Authorization: Bearer {authtoken}"]
        else:
            cmd_args = ["npx", "-y", "mcp-remote", url]
        self.protocol_version = "2025-06-18"
        self.client_name = "mcpo-client"
        
        logger.info(f"Starting mcp-remote subprocess for URL: {url} using protocol {self.protocol_version}")
        
        server_params = StdioServerParameters(
            command=cmd_args[0],
            args=cmd_args[1:],
            env=None
        )
 
        # DISABLED v0.0.62: GitMCP-specific logging (commented out but preserved)
        # Enhanced logging for GitMCP debugging
        # if "gitmcp.io" in url.lower():
        #     logger.info(f"GitMCP command: {' '.join(cmd_args)}")
        #     logger.info("Expected protocol response: 2024-11-05 (not 2025-06-18)")
        #     logger.info("GitMCP initialization starting - this may take 30-45 seconds...")
 
        try:
            async with asyncio.timeout(45):
                self.stdio_context = stdio_client(server_params)
                read_stream, write_stream = await self.stdio_context.__aenter__()
                self.session = ClientSession(read_stream, write_stream)
                
                # Initialize MCP session (let mcp-remote handle protocol negotiation)
                await self.session.initialize()
                logger.info(f"MCP session initialized for {url}")
                
                logger.info(f"mcp-remote connection established successfully using {self.protocol_version}")
                
        except asyncio.TimeoutError:
            # DISABLED v0.0.62: GitMCP-specific timeout handling (commented out but preserved)
            # logger.warning(f"GitMCP initialization timed out after 45 seconds - continuing without GitMCP")
            logger.warning(f"MCP initialization timed out after 45 seconds")
            await self._cleanup_failed_connection()
            raise RuntimeError("MCP connection timeout")
        except Exception as e:
            logger.error(f"Failed to establish mcp-remote connection: {e}")
            await self._cleanup_failed_connection()
            raise

    async def _cleanup_failed_connection(self):
        """Clean up failed connection resources"""
        try:
            if hasattr(self, 'stdio_context') and self.stdio_context:
                await self.stdio_context.__aexit__(None, None, None)
        except Exception as e:
            logger.debug(f"Error during cleanup: {e}")
        finally:
            self.stdio_context = None
            self.session = None
            
    async def stop(self):
        """Stop mcp-remote subprocess"""
        if hasattr(self, 'session') and self.session:
            try:
                await self.session.close()
            except Exception as e:
                logger.warning("Error closing MCP session: %s", e)
        
        if hasattr(self, 'stdio_context'):
            try:
                await self.stdio_context.__aexit__(None, None, None)
            except Exception as e:
                logger.warning("Error closing stdio context: %s", e)
        
        if hasattr(self, 'process') and self.process:
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
        self.stdio_context = None

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

            # FIXED v0.0.44: Consistent empty response handling
            if not content:
                return {"result": ""}

            first_item = content[0]
            if hasattr(first_item, 'text'):
                text = first_item.text
            elif isinstance(first_item, dict):
                text = first_item.get('text', '')
            else:
                text = str(first_item)

            if text:
                try:
                    parsed = json.loads(text)
                    return {"result": parsed}
                except Exception:
                    return {"result": text}

            # Return full content array
            return {"result": content}

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
            # FIXED v0.0.44: Unified error response structure
            detail = {"error": getattr(me, "message", str(me))}
            try:
                if hasattr(me, "args") and me.args:
                    detail["error_details"] = [str(a) for a in me.args]
            except Exception:
                pass
            raise HTTPException(status_code=400, detail=detail)
        except RuntimeError as re:
            return {"error": str(re)}
        except PydValidationError as e:
            raise HTTPException(status_code=502, detail={"error": f"Validation error: {str(e)}"})

        result = resp or {}
        try:
            if isinstance(result, dict):
                if "error" in result and result["error"]:
                    raise HTTPException(status_code=502, detail={"error": result["error"]})
                if "errors" in result and result["errors"]:
                    raise HTTPException(status_code=502, detail={"error": result["errors"]})
        except HTTPException:
            raise

        # FIXED v0.0.67: Handle both dict and Pydantic CallToolResult objects
        if isinstance(result, dict):
            content = result.get("content") or []
        elif hasattr(result, "content"):
            content = getattr(result, "content", []) or []
        else:
            content = []

        # FIXED v0.0.44: Consistent response handling
        if not content:
            return {"result": ""}

        # Preserve content array structure for Open WebUI
        first = content[0] if content else {}
        
        # Handle both dict and Pydantic content items
        if isinstance(first, dict):
            text = first.get("text")
        elif hasattr(first, "text"):
            text = getattr(first, "text", None)
        else:
            text = None

        if text:
            try:
                parsed = json.loads(text)
                return {"result": parsed}
            except Exception:
                return {"result": text}

        # Return full content array
        return {"result": content}

_DISCOVERED_TOOL_NAMES: List[str] = []
_DISCOVERED_TOOLS_MIN: List[Dict[str, Any]] = []

# ENHANCED v0.0.43: Multi-server tool route mounting with Open WebUI compatibility
async def _mount_multi_server_tool_routes(tools: List[ToolDef], servers: List[MCPServerConfig], app: FastAPI):
    """Mount routes for multi-server tools with proper server routing and Open WebUI compatibility"""

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

        # ENHANCED v0.0.43: Create POST handler with Open WebUI request processing
        async def create_post_handler(tool_obj: ToolDef, server: MCPServerConfig, orig_name: str):
            async def handler(request: Request, payload: Dict[str, Any] = Depends(parse_optional_json_body), dep=Depends(api_dependency())):
                try:
                    # ADDED v0.0.43: Enhanced Open WebUI payload processing
                    processed_payload = await _process_open_webui_payload(request, payload)

                    # ADDED v0.0.45: Ensure request compatibility with tool schema
                    tool_schema = tool_obj.inputSchema if tool_obj else {}
                    compatible_payload = _ensure_request_compatibility(processed_payload, tool_schema)

                    logger.debug("Processed payload for %s: %s -> %s -> %s",
                                orig_name, type(payload).__name__, type(processed_payload).__name__, type(compatible_payload).__name__)

                    # Route to specific server with processed payload
                    result = await _call_multi_server_tool(server, orig_name, compatible_payload)
                    return JSONResponse(status_code=200, content=result)

                except HTTPException:
                    raise
                except Exception as e:
                    logger.exception("Multi-server tool call failed for %s on %s: %s", orig_name, server.name, e)
                    raise HTTPException(status_code=502, detail=str(e))
            return handler

        handler = await create_post_handler(tool, server_config, original_tool_name)
        app.post(route_path, name=f"tool_{tool.name}", tags=["tools"])(handler)

        # DISABLED v0.0.62: GitMCP-specific GET route logic (commented out but preserved)
        # ENHANCED v0.0.42: Add GET route for GitMCP tools (always) and parameter-less tools (existing logic)
        # input_schema = tool.inputSchema or {}
        # is_parameter_less = not input_schema.get('properties') and not input_schema.get('required')
        # is_gitmcp_tool = server_name == "gitmcp"
        #
        # if is_parameter_less or is_gitmcp_tool:
        #     async def create_get_handler(tool_obj: ToolDef, server: MCPServerConfig, orig_name: str):
        #         async def get_handler(dep=Depends(api_dependency())):
        #             try:
        #                 # GitMCP tools always use empty payload, parameter-less tools use empty dict
        #                 payload = {} if is_gitmcp_tool else {}
        #                 result = await _call_multi_server_tool(server, orig_name, payload)
        #                 return JSONResponse(status_code=200, content=result)
        #             except HTTPException:
        #                 raise
        #             except Exception as e:
        #                 logger.exception("Multi-server GET tool call failed for %s on %s: %s", orig_name, server.name, e)
        #                 raise HTTPException(status_code=502, detail=str(e))
        #         return get_handler
        #
        #     get_handler = await create_get_handler(tool, server_config, original_tool_name)
        #     app.get(route_path, name=f"tool_{tool.name}_get", tags=["tools"])(get_handler)
        #     logger.info("Added GET route for %s tool: %s", "GitMCP" if is_gitmcp_tool else "parameter-less", route_path)

        # ADDED v0.0.62: Simplified GET route logic (GitMCP detection disabled)
        input_schema = tool.inputSchema or {}
        is_parameter_less = not input_schema.get('properties') and not input_schema.get('required')

        if is_parameter_less:
            async def create_get_handler(tool_obj: ToolDef, server: MCPServerConfig, orig_name: str):
                async def get_handler(dep=Depends(api_dependency())):
                    try:
                        payload = {}
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
            logger.info("Added GET route for parameter-less tool: %s", route_path)

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

            async def handler(payload: Dict[str, Any] = Depends(parse_optional_json_body), _tool=tool, _route=route_path, dep=Depends(api_dependency())):
                try:
                    # ENHANCED v0.0.45: Better empty body handling with schema-aware defaults
                    if payload is None:
                        logger.info("No request body provided - using empty dict for Open WebUI compatibility")
                        payload = {}

                    # Ensure compatibility with tool schema
                    tool_schema = _tool.inputSchema if _tool else {}
                    compatible_payload = _ensure_request_compatibility(payload, tool_schema)

                    result = await call_mcp_tool_via_http_fallback(MCP_SERVER_URL, headers, _tool.name, compatible_payload)
                    # FIXED v0.0.44: Apply consistent response formatting
                    formatted_result = _format_tool_response(result)
                    return JSONResponse(status_code=200, content=formatted_result)
                except Exception as e:
                    logger.warning("Direct HTTP failed for tool %s, trying MCP connector: %s", _tool.name, e)
                    try:
                        async with _connector_wrapper(MCP_SERVER_URL) as (reader, writer):
                            result = await call_mcp_tool(reader, writer, _tool.name, compatible_payload)
                            # FIXED v0.0.44: Apply consistent response formatting
                            formatted_result = _format_tool_response(result)
                            return JSONResponse(status_code=200, content=formatted_result)
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
                        # FIXED v0.0.44: Apply consistent response formatting
                        formatted_result = _format_tool_response(result)
                        return JSONResponse(status_code=200, content=formatted_result)
                    except Exception as e:
                        logger.warning(
                            "Direct HTTP failed for tool %s, trying MCP connector: %s", _tool.name, e
                        )
                        try:
                            async with _connector_wrapper(MCP_SERVER_URL) as (reader, writer):
                                result = await call_mcp_tool(reader, writer, _tool.name, {})
                                # FIXED v0.0.44: Apply consistent response formatting
                                formatted_result = _format_tool_response(result)
                                return JSONResponse(status_code=200, content=formatted_result)
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
            if STDIOAVAILABLE:
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
                                    # FIXED v0.0.45: Accept optional payload to handle empty Open WebUI requests
                                    async def handler(payload: Dict[str, Any] = Depends(parse_optional_json_body), dep=Depends(api_dependency())):
                                        try:
                                            # Handle empty request bodies from Open WebUI
                                            if payload is None:
                                                logger.info("No request body provided for mcp-remote tool %s - using empty dict", tool_name)
                                                payload = {}

                                            result = await manager.call_tool(tool_name, payload or {})
                                            # FIXED v0.0.44: Apply consistent response formatting
                                            formatted_result = _format_tool_response(result)
                                            return JSONResponse(status_code=200, content=formatted_result)
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

        # ADDED v0.0.62: GitMCP status notification
        logger.info("GitMCP integration: DISABLED (v0.0.62 - protocol compatibility issues)")
        logger.info("Context7 integration: PENDING (via Open WebUI Pipelines)")

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
                    # ADDED v0.0.62: Log GitMCP skip status
                    if _is_gitmcp_server(server.name, server.url):
                        logger.info(" - %s: %s (SKIPPED - GitMCP disabled in v0.0.62)", server.name, server.url)
                    else:
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

    # 🔧 CRITICAL FIX: Force all request bodies to be optional for Open WebUI compatibility
    paths = schema.setdefault("paths", {})
    for path, methods in paths.items():
        for method, operation in methods.items():
            if method.lower() == "post" and "requestBody" in operation:
                # Force request body to be optional for Open WebUI compatibility
                operation["requestBody"]["required"] = False
                
                # Ensure content schema allows empty objects
                content = operation["requestBody"].get("content", {})
                for media_type, schema_def in content.items():
                    if "schema" in schema_def:
                        # Allow empty objects and null values
                        schema_def["schema"]["additionalProperties"] = True
                        if "required" in schema_def["schema"]:
                            # Make all properties optional
                            schema_def["schema"]["required"] = []

    # Add time tool to schema if not present
    paths.setdefault("/tools/time", {}).setdefault("post", {}).update({
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
        "tags": ["tools"]
    })

    # Ensure security scheme is properly configured
    components = schema.setdefault("components", {})
    components.setdefault("securitySchemes", {})["apiKeyAuth"] = {
        "type": "apiKey",
        "in": "header", 
        "name": "x-api-key"
    }

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
        "mcp_remote_fallback": "available" if STDIOAVAILABLE else "disabled (StdioClientTransport not found)",
    }

    # ADDED v0.0.62: GitMCP status in diagnostics
    info["gitmcp_status"] = "disabled (v0.0.62 - protocol compatibility issues)"
    info["gitmcp_alternative"] = "Context7 via Open WebUI Pipelines (pending)"
    
    # ADDED v0.0.66: Connection method tracking diagnostics
    info["connection_method_tracking"] = "enabled (v0.0.66)"
    info["tracked_server_methods"] = dict(SERVER_CONNECTION_METHODS)

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
                        "has_auth_token": bool(s.auth_token),
                        # ADDED v0.0.62: GitMCP detection status
                        "is_gitmcp": _is_gitmcp_server(s.name, s.url),
                        "status": "skipped (GitMCP disabled)" if _is_gitmcp_server(s.name, s.url) else "active",
                        # ADDED v0.0.66: Connection method tracking per server
                        "connection_method": SERVER_CONNECTION_METHODS.get(s.name, "not_tracked")
                    } for s in servers
                ]
                info["server_count"] = len(servers)
                info["enabled_server_count"] = sum(1 for s in servers if s.enabled and not _is_gitmcp_server(s.name, s.url))
                info["skipped_gitmcp_count"] = sum(1 for s in servers if _is_gitmcp_server(s.name, s.url))
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
