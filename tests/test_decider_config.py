import asyncio
import os
import sys
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from scripts._decider import (
    DECISIONS_URL,
    MODEL,
    call_decider,
    decider_api_key,
    decider_model,
    decider_url,
)
from scripts._numeric_values import numeric_request
from scripts.area_filter_probe import filter_request


class DeciderConfigTests(unittest.TestCase):
    def test_existing_openrouter_configuration_still_works(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "router-key"}, clear=True):
            self.assertEqual(decider_url(), DECISIONS_URL)
            self.assertEqual(decider_model(), MODEL)
            self.assertEqual(decider_api_key(), "router-key")

    def test_local_provider_does_not_require_or_inherit_openrouter_credentials(self):
        with patch.dict(os.environ, {
            "DECIDER_URL": "http://localhost:8321/v1/systemone",
            "OPENROUTER_API_KEY": "router-only-key",
        }, clear=True):
            self.assertIsNone(decider_api_key())
        with patch.dict(os.environ, {"DECIDER_URL": "http://localhost:8321/v1/systemone"}, clear=True):
            self.assertIsNone(decider_api_key())
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
            decider_api_key()

    def test_generic_provider_key_overrides_legacy_cloud_key(self):
        with patch.dict(os.environ, {
            "DECIDER_URL": "https://example.test/v1/systemone", "DECIDER_API_KEY": "provider-key",
            "OPENROUTER_API_KEY": "router-only-key",
        }, clear=True):
            self.assertEqual(decider_api_key(), "provider-key")

    def test_model_override_reaches_area_filter_and_numeric_probes(self):
        with patch.dict(os.environ, {"DECIDER_MODEL": "local-decider"}, clear=True):
            _, area = filter_request("Turn off the lights", "en", None)
            _, numeric = numeric_request("Set brightness to 50", "brightness", 20, 0, 100, 6, "%", model=decider_model())
            self.assertEqual([area["model"], numeric["model"]], ["local-decider"] * 2)

    def test_invalid_url_is_reported(self):
        for url in ("localhost:8321", "file:///tmp/model", "https://"):
            with self.subTest(url=url), patch.dict(os.environ, {"DECIDER_URL": url}, clear=True):
                with self.assertRaisesRegex(ValueError, "full HTTP"):
                    decider_url()

    def test_native_payload_and_response_use_configured_endpoint_with_optional_auth(self):
        for authenticated in (False, True):
            data = {"model": "local-decider", "answers": {}}
            response = SimpleNamespace(is_error=False, json=lambda: data)
            client = SimpleNamespace(post=AsyncMock(return_value=response))
            @asynccontextmanager
            async def connection(**kwargs):
                yield client
            fake_httpx = SimpleNamespace(AsyncClient=connection)
            environment = {
                "DECIDER_URL": "http://localhost:8321/v1/systemone",
                "OPENROUTER_API_KEY": "router-only-key",
            }
            if authenticated:
                environment["DECIDER_API_KEY"] = "local-key"
            payload = {"model": "local-decider", "state": {"prompt": "test"}, "questions": {}}
            with self.subTest(authenticated=authenticated), patch.dict(os.environ, environment, clear=True), \
                    patch.dict(sys.modules, {"httpx": fake_httpx}):
                self.assertEqual(asyncio.run(call_decider(payload, decider_api_key())), data)
            client.post.assert_awaited_once_with(
                "http://localhost:8321/v1/systemone", json=payload,
                headers={"Authorization": "Bearer local-key"} if authenticated else {},
            )

    def test_provider_http_error_has_neutral_message(self):
        response = SimpleNamespace(is_error=True, status_code=501, text="Decision model required")
        client = SimpleNamespace(post=AsyncMock(return_value=response))
        @asynccontextmanager
        async def connection(**kwargs):
            yield client
        with patch.dict(os.environ, {"DECIDER_URL": "http://localhost:8321/v1/systemone"}, clear=True), \
                patch.dict(sys.modules, {"httpx": SimpleNamespace(AsyncClient=connection)}):
            with self.assertRaisesRegex(ValueError, "Decider HTTP 501"):
                asyncio.run(call_decider({"state": "test", "questions": {}}))


if __name__ == "__main__":
    unittest.main()
