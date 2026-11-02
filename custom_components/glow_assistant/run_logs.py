import asyncio
import json
import math
import os
import re
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

_SECRET_KEYS = {"apikey", "token", "accesstoken", "refreshtoken", "bearertoken", "password", "authorization",
                "proxyauthorization", "cookie", "setcookie", "headers"}


def redact(value, secrets=()):
    if isinstance(value, dict):
        return {key: "[redacted]" if re.sub(r"[^a-z]", "", str(key).lower()) in _SECRET_KEYS else redact(item, secrets)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in sorted(secrets, key=len, reverse=True):
            if secret:
                value = value.replace(secret, "[redacted]")
        value = re.sub(r"(?i)\bBearer\s+[^\s\"'<>]+", "Bearer [redacted]", value)
        value = re.sub(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@", r"\1[redacted]@", value)
        return re.sub(r"(?i)([?&](?:api[_-]?key|access_token|token|password|key)=)[^&#\s]+", r"\1[redacted]", value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def successful(result):
    return (result.get("route") in ("smarthome", "not-smarthome-related") and isinstance(result.get("answer"), str) and bool(result["answer"].strip())
            and "error" not in result and "answer_error" not in result and not result.get("failures")
            and (not result.get("execution") or result["execution"]["status"] == "completed"
                 or result["execution"]["status"] == "planned" and result["execution"].get("dry_run")))


def _write(path, report):
    manifest = Path(__file__).with_name("manifest.json")
    report["integration_version"] = json.loads(manifest.read_text(encoding="utf-8"))["version"]
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".run-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, default=str)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class RunRecord:
    def __init__(self, source, prompt, language, *, dry_run=False, enabled=False, capture=True, capture_models=False,
                 trace=None, secrets=()):
        self.enabled = enabled is True
        self.capture = capture or enabled
        self.capture_models = capture_models or enabled
        self.trace_callback = trace
        self.stage = "context"
        self.secrets = set(secret for secret in secrets if secret)
        self.started = perf_counter()
        self.data = {"schema_version": 1, "run_id": uuid4().hex, "source": source,
                     "mode": "plan_only" if dry_run else "live", "started_at": datetime.now(timezone.utc).isoformat(),
                     "language": language, "request": prompt, "context": {"entities": 0, "actions": 0},
                     "trace": [], "model_calls": []}

    def settings(self, settings):
        self.enabled = settings["log_runs"] is True
        self.capture = self.capture or self.enabled
        self.capture_models = self.capture_models or self.enabled
        credentials = [settings[group].get("api_key") for group in ("decider", "answer", "general")]
        servers = settings["general"]["mcp_servers"]
        if isinstance(servers, list):
            credentials.extend(server.get("token") for server in servers if isinstance(server, dict))
        self.secrets.update(value for value in credentials if isinstance(value, str) and value)
        if self.capture:
            self.data["settings"] = redact(deepcopy(settings), self.secrets)

    def snapshot(self, snapshot):
        self.data["context"] = {"entities": len(snapshot["entities"]), "actions": len(snapshot["services"])}
        if self.capture:
            self.data["snapshot"] = snapshot

    def trace(self, stage, arguments, result=None):
        if result is None and stage != "failure":
            self.stage = stage
        if self.capture:
            event = {"stage": stage, "offset_ms": round((perf_counter() - self.started) * 1000)}
            event["input" if result is None else "output"] = arguments if result is None else result
            self.data["trace"].append(deepcopy(event))
        if self.trace_callback:
            self.trace_callback(stage, redact(arguments, self.secrets),
                                redact(result, self.secrets) if result is not None else None)

    def model_call(self, call):
        if self.capture_models:
            call["stage"] = self.stage
            call["offset_ms"] = round((perf_counter() - self.started) * 1000)
            self.data["model_calls"].append(call)

    async def finish(self, result, directory, **metadata):
        self.data.update(metadata, result=result, success=successful(result), finished_at=datetime.now(timezone.utc).isoformat())
        self.data.setdefault("timings_ms", {"total": round((perf_counter() - self.started) * 1000)})
        try:
            report = redact(self.data, self.secrets)
        except Exception as error:
            return {"result": result, "success": successful(result),
                    "log_error": f"Could not prepare run log ({type(error).__name__})."}
        if self.enabled:
            try:
                category = "successful" if report["success"] else "unsuccessful"
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                path = Path(directory) / category / f"{timestamp}-{report['run_id']}.json"
                report["log_file"] = str(path)
                await asyncio.to_thread(_write, path, report)
            except Exception as error:
                report.pop("log_file", None)
                report["log_error"] = redact(str(error), self.secrets)
        return report
