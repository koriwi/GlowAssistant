import asyncio
import json
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta

from .client import ModelError


@asynccontextmanager
async def connect(server):
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    headers = {"Authorization": f"Bearer {server['token']}"} if server["token"] else {}
    async with (
        httpx.AsyncClient(headers=headers, timeout=server["timeout"], follow_redirects=False) as client,
        streamable_http_client(server["url"], http_client=client) as streams,
        ClientSession(streams[0], streams[1], read_timeout_seconds=timedelta(seconds=server["timeout"])) as session,
    ):
        await session.initialize()
        yield session


class MCPTools:
    def __init__(self, servers, max_result_chars, connection=connect):
        self.servers = [server for server in servers if server["enable"]]
        self.max_result_chars = max_result_chars
        self.connection = connection
        self.stack = AsyncExitStack()
        self.tools = []
        self.targets = {}

    async def __aenter__(self):
        try:
            for server_index, server in enumerate(self.servers, 1):
                session = await self.stack.enter_async_context(self.connection(server))
                cursor = None
                seen = set()
                available = set()
                tool_index = 0
                while True:
                    page = await session.list_tools(cursor=cursor)
                    for tool in page.tools:
                        available.add(tool.name)
                        if server["tools"] and tool.name not in server["tools"]:
                            continue
                        tool_index += 1
                        name = f"mcp_{server_index:03}_{tool_index:03}"
                        self.tools.append({"type": "function", "function": {
                            "name": name,
                            "description": f"{server['name']}: {tool.name}. {tool.description or ''}",
                            "parameters": tool.inputSchema,
                        }})
                        self.targets[name] = (server, session, tool.name, tool.inputSchema)
                    cursor = page.nextCursor
                    if not cursor:
                        break
                    if cursor in seen:
                        raise ModelError("MCP tool discovery returned a repeated cursor.")
                    seen.add(cursor)
                if set(server["tools"]) - available:
                    raise ModelError("Configured MCP tools are not advertised by the server.")
            return self
        except BaseException as error:
            try:
                await self.stack.aclose()
            except Exception as cleanup_error:
                raise ModelError("MCP connection cleanup failed.") from cleanup_error
            if isinstance(error, ModelError):
                raise
            if isinstance(error, Exception):
                raise ModelError("MCP server connection or tool discovery failed.") from error
            raise

    async def __aexit__(self, *args):
        try:
            await self.stack.aclose()
        except Exception as error:
            if args[0] is None:
                raise ModelError("MCP session cleanup failed; tool calls were not retried.") from error
        return False

    def describe(self, name):
        target = self.targets.get(name)
        if target is None:
            raise ModelError("General model requested an unknown or disabled MCP tool.")
        server, _, tool_name, _ = target
        return {"server": server["name"], "tool": tool_name}

    async def call(self, name, arguments):
        self.describe(name)
        from jsonschema import SchemaError, ValidationError, validate

        server, session, tool_name, schema = self.targets[name]
        try:
            validate(arguments, schema)
        except (SchemaError, ValidationError) as error:
            raise ModelError("MCP arguments do not match the advertised tool schema; no call was made.") from error
        try:
            async with asyncio.timeout(server["timeout"]):
                response = await session.call_tool(tool_name, arguments)
        except Exception as error:
            raise ModelError("MCP tool call failed; its outcome is unconfirmed. It was not retried.") from error
        result = response.model_dump(mode="json", exclude_none=True)
        if len(json.dumps(result, ensure_ascii=False)) > self.max_result_chars:
            raise ModelError("MCP tool result exceeds the configured character limit; the tool was not retried.")
        return result
