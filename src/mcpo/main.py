# main.py_v0.0.27

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


# -------------------------------------------------------------------
# validate_server_config, load_config, create_sub_app,
# mount_config_servers, unmount_servers, reload_config_handler
# remain unchanged from v0.0.26
# -------------------------------------------------------------------


class TimeResponse(BaseModel):
    now_utc: str


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

                # --- Force extraction from TextContent ---
                try:
                    if result and getattr(result, "content", None):
                        first = result.content[0]
                        if hasattr(first, "text") and first.text:
                            return {"now_utc": first.text}
                except Exception as e:
                    logger.warning(f"/time force extract failed: {e}")

                # If dict (rare but supported)
                if isinstance(result, dict) and "now_utc" in result:
                    return result

                if isinstance(result, str):
                    return {"now_utc": result}

                # Final fallback
                return {"now_utc": str(result)}
                # -----------------------------------------

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


# -------------------------------------------------------------------
# lifespan(), run(), echo/ping routes are identical to v0.0.26
# (unchanged, and confirmed working)
# -------------------------------------------------------------------
