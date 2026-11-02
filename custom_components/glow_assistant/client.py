import json
from copy import deepcopy
from time import perf_counter

import aiohttp


class ModelError(Exception):
    def __init__(self, message, **details):
        super().__init__(message)
        self.details = details


class ModelClient:
    def __init__(self, session, observer=None):
        self.session = session
        self.observer = observer

    async def _post(self, config, payload):
        if self.observer is None:
            return await self._request(config, payload)
        call = {"endpoint": config["url"], "settings": deepcopy({key: value for key, value in config.items() if key != "api_key"}),
                "request": deepcopy(payload), "status": "pending"}
        self.observer(call)
        started = perf_counter()
        try:
            response = await self._request(config, payload, call)
            call["status"] = "response_received"
            return response
        except Exception as error:
            call.update(status="error", error=str(error), error_type=type(error).__name__, details=getattr(error, "details", {}))
            raise
        finally:
            call["duration_ms"] = round((perf_counter() - started) * 1000)

    async def _request(self, config, payload, call=None):
        body = json.dumps(payload, ensure_ascii=False)
        input_chars = len(body)
        limit = config["max_input_chars"]
        if input_chars > limit:
            raise ModelError(
                f"Model input exceeds its configured character limit: {input_chars:,} characters; configured limit: {limit:,}.",
                input_chars=input_chars, max_input_chars=limit, model=config["model"],
            )
        headers = {"Authorization": f"Bearer {config['api_key']}"} if config["api_key"] else {}
        try:
            async with self.session.post(
                config["url"], json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=config["timeout"]), allow_redirects=False,
            ) as response:
                if call is not None:
                    call["http_status"] = response.status
                if response.status != 200:
                    if call is not None:
                        try:
                            call["response"] = deepcopy(await response.json(content_type=None))
                        except Exception:
                            call["response_error"] = "The HTTP error body was not available as JSON."
                    raise ModelError(f"Model endpoint returned HTTP {response.status}.")
                data = await response.json(content_type=None)
                if call is not None:
                    call["response"] = deepcopy(data)
        except TimeoutError as error:
            raise ModelError("Model request timed out.") from error
        except aiohttp.ClientError as error:
            raise ModelError("Model connection failed.") from error
        except ValueError as error:
            raise ModelError("Model endpoint returned invalid JSON.") from error
        if not isinstance(data, dict) or "error" in data:
            raise ModelError("Model endpoint returned an error or invalid object.")
        return data

    async def decide(self, config, state, questions):
        return await self._post(config, {
            "model": config["model"], "state": state, "questions": questions,
        })

    async def chat(self, config, messages):
        message, usage = await self.completion(config, messages)
        if message.get("tool_calls"):
            raise ModelError("Unexpected tool calls from a model without tools.")
        return message["content"].strip(), usage

    async def completion(self, config, messages, tools=None):
        payload = {
            "model": config["model"], "messages": messages,
            "max_tokens": config["max_tokens"], "temperature": config["temperature"],
        }
        if tools:
            payload.update(tools=tools, tool_choice="auto", parallel_tool_calls=False)
        data = await self._post(config, payload)
        try:
            choice = data["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ModelError("Model reply reached its output token limit.")
            message = choice["message"]
            text = message.get("content")
            calls = message.get("tool_calls")
            if calls is not None and not isinstance(calls, list):
                raise ModelError("Model returned malformed tool calls.")
            if not calls and (not isinstance(text, str) or not text.strip()):
                raise ModelError("Model returned no spoken answer.")
        except (KeyError, IndexError, TypeError, AttributeError) as error:
            raise ModelError("Model returned a malformed chat response.") from error
        return {**message, "role": "assistant"}, data.get("usage", {})
