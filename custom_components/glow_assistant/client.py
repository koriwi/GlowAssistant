import json

import aiohttp


class ModelError(Exception):
    pass


class ModelClient:
    def __init__(self, session):
        self.session = session

    async def _post(self, config, payload):
        body = json.dumps(payload, ensure_ascii=False)
        if len(body) > config["max_input_chars"]:
            raise ModelError("Model input exceeds its configured character limit.")
        headers = {"Authorization": f"Bearer {config['api_key']}"} if config["api_key"] else {}
        try:
            async with self.session.post(
                config["url"], json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=config["timeout"]), allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise ModelError(f"Model endpoint returned HTTP {response.status}.")
                data = await response.json(content_type=None)
        except (aiohttp.ClientError, TimeoutError, ValueError) as error:
            raise ModelError("Model connection failed or returned invalid JSON.") from error
        if not isinstance(data, dict) or "error" in data:
            raise ModelError("Model endpoint returned an error or invalid object.")
        return data

    async def decide(self, config, state, questions):
        return await self._post(config, {
            "model": config["model"], "state": state, "questions": questions,
        })

    async def chat(self, config, messages):
        data = await self._post(config, {
            "model": config["model"], "messages": messages,
            "max_tokens": config["max_tokens"], "temperature": config["temperature"],
        })
        try:
            choice = data["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ModelError("Model reply reached its output token limit.")
            text = choice["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise ModelError("Model returned no spoken answer.")
        except (KeyError, IndexError, TypeError, AttributeError) as error:
            raise ModelError("Model returned a malformed chat response.") from error
        return text.strip(), data.get("usage", {})
