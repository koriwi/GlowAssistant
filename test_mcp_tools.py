import asyncio
import importlib.util
import json
import sys
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from custom_components.glow_assistant.client import ModelError
from custom_components.glow_assistant.mcp_tools import MCPTools, connect

SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}


def server(name="search", **updates):
    return {"name": name, "url": "http://mcp.example/mcp", "token": "", "timeout": 30, "tools": [], "enable": True, **updates}


def tool(name="search"):
    return SimpleNamespace(name=name, description="Find information", inputSchema=SCHEMA)


def session(tools=None, result=None):
    data = result if result is not None else {"isError": False, "content": [{"type": "text", "text": "Found"}]}
    response = SimpleNamespace(model_dump=lambda **kwargs: data)
    return SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=tools or [tool()], nextCursor=None)),
        call_tool=AsyncMock(return_value=response),
    )


class MCPToolsTests(unittest.IsolatedAsyncioTestCase):
    def connection(self, sessions):
        self.opened = []
        self.closed = []
        @asynccontextmanager
        async def connection(config):
            self.opened.append(config["name"])
            try:
                yield sessions[config["name"]]
            finally:
                self.closed.append(config["name"])
        return connection

    async def test_two_servers_with_duplicate_tool_names_remain_separate(self):
        first, second = session(), session()
        connection = self.connection({"first": first, "second": second})
        async with MCPTools([server("first"), server("second")], 2000, connection) as tools:
            self.assertEqual([item["function"]["name"] for item in tools.tools], ["mcp_001_001", "mcp_002_001"])
            await tools.call("mcp_002_001", {"query": "weather"})
            self.assertEqual(tools.describe("mcp_002_001"), {"server": "second", "tool": "search"})
        second.call_tool.assert_awaited_once_with("search", {"query": "weather"})
        first.call_tool.assert_not_awaited()
        self.assertEqual(self.closed, ["second", "first"])

    async def test_disabled_servers_and_disallowed_tools_are_not_exposed(self):
        allowed = session([tool("find"), tool("write")])
        connection = self.connection({"allowed": allowed})
        async with MCPTools([server("disabled", enable=False), server("allowed", tools=["find"])], 2000, connection) as tools:
            self.assertEqual(len(tools.tools), 1)
            self.assertEqual(tools.describe("mcp_001_001")["tool"], "find")
            with self.assertRaisesRegex(ModelError, "unknown or disabled"):
                await tools.call("write", {})
        allowed.call_tool.assert_not_awaited()
        self.assertEqual(self.opened, ["allowed"])

    async def test_discovery_reads_all_pages_once(self):
        target = session()
        target.list_tools.side_effect = [SimpleNamespace(tools=[tool("first")], nextCursor="next"),
                                         SimpleNamespace(tools=[tool("second")], nextCursor=None)]
        async with MCPTools([server()], 2000, self.connection({"search": target})) as tools:
            self.assertEqual(len(tools.tools), 2)
        self.assertEqual(target.list_tools.await_count, 2)
        self.assertEqual(target.list_tools.await_args_list[1].kwargs, {"cursor": "next"})

    async def test_invalid_arguments_are_rejected_before_execution(self):
        target = session()
        async with MCPTools([server()], 2000, self.connection({"search": target})) as tools:
            for arguments in ({}, {"query": 123}, {"query": "x", "extra": True}):
                with self.subTest(arguments=arguments), self.assertRaisesRegex(ModelError, "schema"):
                    await tools.call("mcp_001_001", arguments)
        target.call_tool.assert_not_awaited()

    async def test_tool_allowlist_with_unknown_names_is_a_discovery_error(self):
        target = session()
        with self.assertRaisesRegex(ModelError, "not advertised"):
            async with MCPTools([server(tools=["unknown"])], 2000, self.connection({"search": target})):
                self.fail("Unknown tools should not be ignored")
        target.call_tool.assert_not_awaited()

    async def test_transport_failure_is_not_retried(self):
        target = session()
        target.call_tool.side_effect = TimeoutError("private-url?token=secret")
        async with MCPTools([server()], 2000, self.connection({"search": target})) as tools:
            with self.assertRaisesRegex(ModelError, "outcome is unconfirmed") as caught:
                await tools.call("mcp_001_001", {"query": "x"})
        self.assertNotIn("secret", str(caught.exception))
        target.call_tool.assert_awaited_once()

    async def test_large_results_are_not_retried_or_silently_truncated(self):
        target = session(result={"content": [{"type": "text", "text": "x" * 500}]})
        async with MCPTools([server()], 100, self.connection({"search": target})) as tools:
            with self.assertRaisesRegex(ModelError, "character limit"):
                await tools.call("mcp_001_001", {"query": "x"})
        target.call_tool.assert_awaited_once()

    async def test_body_error_is_preserved_and_connections_close(self):
        with self.assertRaisesRegex(ModelError, "Original error"):
            async with MCPTools([server()], 2000, self.connection({"search": session()})):
                raise ModelError("Original error")
        self.assertEqual(self.closed, ["search"])

    async def test_discovery_failure_closes_previous_connections_and_hides_credentials(self):
        failing = session()
        failing.list_tools.side_effect = ValueError("http://private?token=secret")
        with self.assertRaisesRegex(ModelError, "discovery failed") as caught:
            async with MCPTools([server("first"), server("failed")], 2000,
                                self.connection({"first": session(), "failed": failing})):
                self.fail("Discovery should fail")
        self.assertNotIn("secret", str(caught.exception))
        self.assertEqual(self.closed, ["failed", "first"])

    async def test_http_transport_uses_only_the_server_token(self):
        target = SimpleNamespace(initialize=AsyncMock())
        client = object()
        @asynccontextmanager
        async def http_connection(**kwargs):
            yield client
        @asynccontextmanager
        async def transport(url, **kwargs):
            yield ("read", "write", None)
        @asynccontextmanager
        async def sdk_session(*args, **kwargs):
            yield target
        client_factory = Mock(side_effect=http_connection)
        transport_factory = Mock(side_effect=transport)
        session_factory = Mock(side_effect=sdk_session)
        dependencies = {
            "httpx": SimpleNamespace(AsyncClient=client_factory),
            "mcp": SimpleNamespace(ClientSession=session_factory),
            "mcp.client.streamable_http": SimpleNamespace(streamable_http_client=transport_factory),
        }
        with patch.dict(sys.modules, dependencies):
            async with connect(server(token="mcp-only-key")) as result:
                self.assertIs(result, target)
        client_factory.assert_called_once_with(headers={"Authorization": "Bearer mcp-only-key"}, timeout=30, follow_redirects=False)
        transport_factory.assert_called_once_with("http://mcp.example/mcp", http_client=client)
        target.initialize.assert_awaited_once()


@unittest.skipUnless(importlib.util.find_spec("mcp"), "MCP SDK not installed; run the CI suite")
class MCPHTTPContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_sdk_connects_to_a_local_http_mcp_server(self):
        import socket

        import uvicorn
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("test", stateless_http=True, json_response=True)
        @mcp.tool()
        def add(a: int, b: int) -> int:
            return a + b
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        web = uvicorn.Server(uvicorn.Config(mcp.streamable_http_app(), log_level="error", lifespan="on"))
        task = asyncio.create_task(web.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(10):
                while not web.started:
                    if task.done():
                        await task
                    await asyncio.sleep(0.01)
            async with MCPTools([server(url=f"http://127.0.0.1:{port}/mcp")], 2000) as tools:
                result = await tools.call(tools.tools[0]["function"]["name"], {"a": 3, "b": 4})
                self.assertFalse(result.get("isError"))
                self.assertIn("7", json.dumps(result))
        finally:
            web.should_exit = True
            await asyncio.wait_for(task, 10)
            listener.close()


if __name__ == "__main__":
    unittest.main()
