import asyncio
import json
import logging
import os
import signal
import socket
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Optional, Dict, Any
from urllib.parse import urljoin

import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.routing import Mount

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from mcpo.utils.auth import APIKeyMiddleware, get_verify_api_key
from mcpo.utils.main import (
    get_model_fields,
    get_tool_handler,
    normalize_server_type,
)
from mcpo.utils.config_watcher import ConfigWatcher

logger = logging.getLogger(__name__)


class GracefulShutdown:
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.tasks = set()

    def handle_signal(self, sig, frame=None):
        logger.info(f"\nReceived {signal.Signals(sig).name}, initiating graceful shutdown...")
        self.shutdown_event.set()

    def track_task(self, task):
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)


def validate_server_config(server_name: str, server_cfg: Dict[str, Any]) -> None:
    server_type = server_cfg.get("type")
    if normalize_server_type(server_type) in ("sse", "streamable-http"):
        if not server_cfg.get("url"):
            raise ValueError(
                f"Server '{server_name}' of type '{server_type}' requires a 'url' field"
            )
    elif server_cfg.get("command"):
        if not isinstance(server_cfg["command"], str):
            raise ValueError(f"Server '{server_name}' 'command' must be a string")
        if server_cfg.get("args") and not isinstance(server_cfg["args"], list):
            raise ValueError(f"Server '{server_name}' 'args' must be a list")
    elif server_cfg.get("url") and not server_type:
        # Fallback for old SSE config without explicit type
        pass
    else:
        raise ValueError(
            f"Server '{server_name}' must have either 'command' for stdio "
            f"or 'type' and 'url' for remote servers"
        )


def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)

        mcp_servers = config_data.get("mcpServers", {})
        if not mcp_servers:
            logger.error(f"No 'mcpServers' found in config file: {config_path}")
            raise ValueError("No 'mcpServers' found in config file.")

        for server_name, server_cfg in mcp_servers.items():
            validate_server_config(server_name, server_cfg)

        return config_data

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file {config_path}: {e}")
        raise
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        raise


def create_sub_app(
    server_name: str,
    server_cfg: Dict[str, Any],
    cors_allow_origins,
    api_key: Optional[str],
    strict_auth: bool,
    api_dependency,
    connection_timeout,
    lifespan,
) -> FastAPI:
    sub_app = FastAPI(
        title=f"{server_name}",
        description=f"{server_name} MCP Server\n\n- [back to tool list](/docs)",
        version="1.0",
        lifespan=lifespan,
    )

    sub_app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_allow_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if server_cfg.get("command"):
        sub_app.state.server_type = "stdio"
        sub_app.state.command = server_cfg["command"]
        sub_app.state.args = server_cfg.get("args", [])
        sub_app.state.env = {**os.environ, **server_cfg.get("env", {})}

    server_config_type = server_cfg.get("type")
    if server_config_type == "sse" and server_cfg.get("url"):
        sub_app.state.server_type = "sse"
        sub_app.state.args = [server_cfg["url"]]
        sub_app.state.headers = server_cfg.get("headers")
    elif normalize_server_type(server_config_type) == "streamable-http" and server_cfg.get("url"):
        url = server_cfg["url"]
        sub_app.state.server_type = "streamablehttp"
        sub_app.state.args = [url]
        sub_app.state.headers = server_cfg.get("headers")
    elif not server_config_type and server_cfg.get("url"):
        sub_app.state.server_type = "sse"
        sub_app.state.args = [server_cfg["url"]]
        sub_app.state.headers = server_cfg.get("headers")

    if api_key and strict_auth:
        sub_app.add_middleware(APIKeyMiddleware, api_key=api_key)

    sub_app.state.api_dependency = api_dependency
    sub_app.state.connection_timeout = connection_timeout

    return sub_app


def mount_config_servers(
    main_app: FastAPI,
    config_data: Dict[str, Any],
    cors_allow_origins,
    api_key: Optional[str],
    strict_auth: bool,
    api_dependency,
    connection_timeout,
    lifespan,
    path_prefix: str,
):
    mcp_servers = config_data.get("mcpServers", {})

    logger.info("Configuring MCP Servers:")
    for server_name, server_cfg in mcp_servers.items():
        sub_app = create_sub_app(
            server_name,
            server_cfg,
            cors_allow_origins,
            api_key,
            strict_auth,
            api_dependency,
            connection_timeout,
            lifespan,
        )
        main_app.mount(f"{path_prefix}{server_name}", sub_app)


def unmount_servers(main_app: FastAPI, path_prefix: str, server_names: list):
    for server_name in server_names:
        mount_path = f"{path_prefix}{server_name}"
        routes_to_remove = []
        for route in main_app.router.routes:
            if hasattr(route, "path") and route.path == mount_path:
                routes_to_remove.append(route)
        for route in routes_to_remove:
            main_app.router.routes.remove(route)
        logger.info(f"Unmounted server: {server_name}")


async def reload_config_handler(main_app: FastAPI, new_config_data: Dict[str, Any]):
    old_config_data = getattr(main_app.state, "config_data", {})
    backup_routes = list(main_app.router.routes)

    try:
        old_servers = set(old_config_data.get("mcpServers", {}).keys())
        new_servers = set(new_config_data.get("mcpServers", {}).keys())

        servers_to_add = new_servers - old_servers
        servers_to_remove = old_servers - new_servers
        servers_to_check = old_servers & new_servers

        cors_allow_origins = getattr(main_app.state, "cors_allow_origins", ["*"])
        api_key = getattr(main_app.state, "api_key", None)
        strict_auth = getattr(main_app.state, "strict_auth", False)
        api_dependency = getattr(main_app.state, "api_dependency", None)
        connection_timeout = getattr(main_app.state, "connection_timeout", None)
        lifespan = getattr(main_app.state, "lifespan", None)
        path_prefix = getattr(main_app.state, "path_prefix", "/")

        if servers_to_remove:
            logger.info(f"Removing servers: {list(servers_to_remove)}")
            unmount_servers(main_app, path_prefix, list(servers_to_remove))

        servers_to_update = []
        for server_name in servers_to_check:
            old_cfg = old_config_data["mcpServers"][server_name]
            new_cfg = new_config_data["mcpServers"][server_name]
            if old_cfg != new_cfg:
                servers_to_update.append(server_name)

        if servers_to_update:
            logger.info(f"Updating servers: {servers_to_update}")
            unmount_servers(main_app, path_prefix, servers_to_update)
            servers_to_add.update(servers_to_update)

        if servers_to_add:
            logger.info(f"Adding servers: {list(servers_to_add)}")
            for server_name in servers_to_add:
                server_cfg = new_config_data["mcpServers"][server_name]
                try:
                    sub_app = create_sub_app(
                        server_name,
                        server_cfg,
                        cors_allow_origins,
                        api_key,
                        strict_auth,
                        api_dependency,
                        connection_timeout,
                        lifespan,
                    )
                    main_app.mount(f"{path_prefix}{server_name}", sub_app)
                except Exception as e:
                    logger.error(f"Failed to create server '{server_name}': {e}")
                    main_app.router.routes = backup_routes
                    raise

        main_app.state.config_data = new_config_data
        logger.info("Config reload completed successfully")

    except Exception as e:
        logger.error(f"Error during config reload, keeping previous configuration: {e}")
        main_app.router.routes = backup_routes
        raise


# --- NEW PATCH: Ensure /time returns proper JSON object ---
class TimeResponse(BaseModel):
    now_utc: str
# ---------------------------------------------------------


async def create_dynamic_endpoints(app: FastAPI, api_dependency=None):
    session: ClientSession = app.state.session
    if not session:
        raise ValueError("Session is not initialized in the app state.")

    result = await session.initialize()
    server_info = getattr(result, "serverInfo", None)
    if server_info:
        app.title = server_info.name or app.title
        app.description = f"{server_info.name} MCP Server" if server_info.name else app.description
        app.version = server_info.version or app.version

    instructions = getattr(result, "instructions", None)
    if instructions:
        app.description = instructions

    tools_result = await session.list_tools()
    tools = tools_result.tools

    for tool in tools:
        endpoint_name = tool.name
        endpoint_description = tool.description
        inputSchema = tool.inputSchema
        outputSchema = getattr(tool, "outputSchema", None)

        form_model_fields = get_model_fields(
            f"{endpoint_name}_form_model",
            inputSchema.get("properties", {}),
            inputSchema.get("required", []),
            inputSchema.get("$defs", {}),
        )

        response_model_fields = None
        if outputSchema:
            response_model_fields = get_model_fields(
                f"{endpoint_name}_response_model",
                outputSchema.get("properties", {}),
                outputSchema.get("required", []),
                outputSchema.get("$defs", {}),
            )

        # SPECIAL-CASE: patched /time
        if endpoint_name == "time":
            async def time_handler(
                req: Request,
                _=Depends(api_dependency) if api_dependency else None,
            ):
                try:
                    body = await req.json()
                except Exception:
                    body = {}
                if not isinstance(body, dict):
                    body = {}

                result = await session.call_tool("time", arguments=body)

                # 🔧 Normalize everything to object form
                if isinstance(result, str):
                    return {"now_utc": result}
                if isinstance(result, dict) and "now_utc" in result:
                    return result
                return {"now_utc": str(result)}

            app.post(
                f"/{endpoint_name}",
                summary="Time",
                description=endpoint_description or "Returns current UTC timestamp.",
                response_model=TimeResponse,
                response_model_exclude_none=True,
                dependencies=[Depends(api_dependency)] if api_dependency else [],
            )(time_handler)
            continue

        # Default handler for other tools
        tool_handler = get_tool_handler(
            session,
            endpoint_name,
            form_model_fields,
            response_model_fields,
        )

        app.post(
            f"/{endpoint_name}",
            summary=endpoint_name.replace("_", " ").title(),
            description=endpoint_description,
            response_model_exclude_none=True,
            dependencies=[Depends(api_dependency)] if api_dependency else [],
        )(tool_handler)


# --- lifespan, run(), echo/ping routes remain unchanged from v0.0.21 ---
# (omitted here for brevity, but copy over exactly from your v0.0.21 file)
