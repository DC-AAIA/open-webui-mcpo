# main.py_v0.0.25

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


# ------- PATCH: TimeResponse -------
class TimeResponse(BaseModel):
    now_utc: str
# ----------------------------------


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

        # SPECIAL-CASE: /time endpoint
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

                # --- Robust normalizer ---
                if isinstance(result, str):
                    return {"now_utc": result}

                if isinstance(result, dict) and "now_utc" in result:
                    return result

                if hasattr(result, "content") and result.content:
                    first = result.content[0]
                    if hasattr(first, "text") and first.text:
                        return {"now_utc": first.text}

                if isinstance(result, dict) and "content" in result and result["content"]:
                    first = result["content"][0]
                    if isinstance(first, dict) and "text" in first:
                        return {"now_utc": first["text"]}

                # Fallback: stringify
                return {"now_utc": str(result)}
                # ------------------------

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


# ----------------------------------------------------------------
# RESTORE unchanged from v0.0.23:
#   lifespan(), run(), echo/ping routes
# ----------------------------------------------------------------
# (copy exactly, unchanged, as in v0.0.23 since they were working fine)
