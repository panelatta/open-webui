"""Display-only generation identity, kept outside provider messages and reasoning."""

import json
import logging
import re
from contextlib import aclosing

from open_webui.utils.reasoning_levels import advertised_levels

log = logging.getLogger(__name__)
ACTION = "generation_config"


def merge_generation_status(history, status):
    """Only our stable status ID is an update; other statuses keep append semantics."""
    history = list(history or [])
    if status.get("action") == ACTION and status.get("id"):
        for index, item in enumerate(history):
            if item.get("action") == ACTION and item.get("id") == status["id"]:
                history[index] = status
                return history
    return [*history, status]


def explicit_effort(payload):
    if not isinstance(payload, dict):
        return None
    for container, key in (
        (payload.get("reasoning"), "effort"),
        (payload, "reasoning_effort"),
        (payload.get("output_config"), "effort"),
    ):
        value = container.get(key) if isinstance(container, dict) else None
        if isinstance(value, str) and value.strip():
            return value
    thinking = payload.get("thinking")
    reasoning = payload.get("reasoning")
    if (isinstance(thinking, dict) and thinking.get("type") == "disabled") or (
        isinstance(reasoning, dict) and reasoning.get("enabled") is False
    ):
        return "none"
    for container, key in ((thinking, "budget_tokens"), (reasoning, "max_tokens")):
        value = container.get(key) if isinstance(container, dict) else None
        if type(value) is int:
            return f"budget:{value}"
    return None


def default_effort(model):
    reasoning = model.get("reasoning") or {}
    thinking = model.get("thinking") or {}
    defaults = model.get("default_parameters") or {}
    candidates = [
        reasoning.get("default_effort") if isinstance(reasoning, dict) else None,
        thinking.get("default_level") if isinstance(thinking, dict) else None,
        model.get("default_reasoning_level"),
        explicit_effort(defaults),
    ]
    levels = advertised_levels(model)
    return next(
        (v for v in candidates if isinstance(v, str) and levels and v in levels), None
    )


def has_no_reasoning(model):
    levels = advertised_levels(model)
    if levels is not None:
        return levels == []
    capabilities = ((model.get("info") or {}).get("meta") or {}).get(
        "capabilities"
    ) or {}
    if capabilities.get("reasoning") is False:
        return True
    supported = model.get("supported_parameters")
    if isinstance(supported, list):
        return not any(p in supported for p in ("reasoning", "reasoning_effort"))
    # No name heuristic invents an effort. The known non-reasoning GPT-4 family
    # can still be identified on minimal catalogs without supported_parameters.
    return bool(re.match(r"^gpt-4(?:[.o-]|$)", model.get("id", "").split("/")[-1]))


class GenerationStatus:
    def __init__(self, emitter, message_id, payload, model):
        self.emitter = emitter
        self.id = f"generation-config:{message_id}"
        self.base_model = payload.get("model") or model.get("id") or "unknown"
        self.requested_base_model = self.base_model
        self.effort = explicit_effort(payload)
        self.source = "request" if self.effort is not None else None
        if self.effort is None:
            self.effort = default_effort(model)
            if self.effort is not None:
                self.source = "provider_default"
            elif has_no_reasoning(model):
                self.effort, self.source = "none", "not_applicable"
        self.done = False
        self.pending = ""

    async def publish(self, done):
        value = (
            self.effort
            if self.effort is not None
            else ("unconfirmed" if done else "confirming…")
        )
        suffix = " (provider default)" if self.source == "provider_default" else ""
        data = {
            "id": self.id,
            "action": ACTION,
            "description": f"{self.base_model} · reasoning: {value}{suffix}",
            "done": done,
            "hidden": False,
            "base_model": self.base_model,
            "requested_base_model": self.requested_base_model,
            "reasoning_effort": self.effort,
            "reasoning_effort_source": self.source
            or ("unavailable" if done else "pending"),
        }
        try:
            await self.emitter({"type": "status", "data": data})
        except Exception:
            # A display failure must not break or retry an upstream generation.
            log.warning("Unable to publish generation status", exc_info=True)
        self.done = done

    async def observe(self, event):
        if self.done or not isinstance(event, dict):
            return
        response = (
            event.get("response") if isinstance(event.get("response"), dict) else event
        )
        reported_model = response.get("model")
        if isinstance(reported_model, str) and reported_model:
            self.base_model = reported_model
        actual = explicit_effort(response)
        if actual is not None:
            self.effort, self.source = actual, "response"
            await self.publish(True)

    async def finish(self):
        if not self.done:
            await self.publish(True)

    async def wrap(self, stream):
        try:
            async with aclosing(stream):
                async for line in stream:
                    if not self.done:
                        text = (
                            line.decode("utf-8", "replace")
                            if isinstance(line, bytes)
                            else line
                        )
                        if isinstance(text, str) and text.startswith("data:"):
                            part = text[5:].strip()
                            if part != "[DONE]":
                                self.pending += part + "\n"
                                try:
                                    event = json.loads(self.pending)
                                except (ValueError, TypeError):
                                    if len(self.pending) > 262144:
                                        self.pending = ""
                                else:
                                    self.pending = ""
                                    await self.observe(event)
                        elif isinstance(text, str) and not text.strip():
                            self.pending = ""
                    yield line
        finally:
            await self.finish()


async def start_generation_status(request, metadata, payload, model):
    if (
        not isinstance(metadata, dict)
        or metadata.get("task")
        or metadata.get("internal")
    ):
        return None
    if not all(metadata.get(k) for k in ("user_id", "chat_id", "message_id")):
        return None
    # Trust only the authenticated chat entry point, not raw API metadata.
    trusted = getattr(request.state, "metadata", None)
    if not isinstance(trusted, dict) or any(
        trusted.get(k) != metadata.get(k) for k in ("user_id", "chat_id")
    ):
        return None
    # Tasks, tools and retry rounds reuse the request; each assistant message
    # gets one status, while multi-model replies and regeneration use new IDs.
    key = (metadata["chat_id"], metadata["message_id"])
    seen = getattr(request.state, "generation_status_messages", None)
    if seen is None:
        seen = set()
        request.state.generation_status_messages = seen
    if key in seen:
        return None
    from open_webui.socket.main import get_event_emitter

    emitter = await get_event_emitter(metadata)
    if emitter is None:
        return None
    seen.add(key)
    status = GenerationStatus(emitter, metadata["message_id"], payload, model)
    await status.publish(False)
    return status
