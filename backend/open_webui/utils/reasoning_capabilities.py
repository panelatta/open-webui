"""Supplement missing provider metadata from the provider's public capability feed."""

import logging
import re

import aiohttp
from aiocache import cached

from open_webui.utils.reasoning_levels import advertised_levels

log = logging.getLogger(__name__)
ABLITERATION_THINKING_URL = "https://docs.abliteration.ai/capabilities/thinking.md"
ABLITERATION_LARGE_MODELS = ("abliterated-model-large-v2", "abliterated-model-large")


def parse_abliteration_levels(document, model_id):
    # Parse the model-specific, distinct modes, not the global alias ladder.
    # Require the advertised count to match; changed/ambiguous prose fails closed.
    pattern = (
        r"\x60"
        + re.escape(model_id)
        + r"\x60 runs in (two|three|four|five|six|seven|eight|nine|ten|[0-9]+)"
        + r" (?:reasoning )?modes? [—–-] (.+?)(?: [—–-] |[.;\n])"
    )
    matches = re.findall(pattern, document)
    if len(matches) != 1:
        return None
    count, description = matches[0]
    numbers = dict(
        zip(
            ("two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"),
            range(2, 11),
        )
    )
    expected = int(count) if count.isdigit() else numbers[count]
    levels = re.findall(r"\*\*([a-zA-Z0-9_-]+)\*\*", description)
    if len(levels) != expected or len(set(levels)) != expected:
        return None
    return advertised_levels({"thinking": {"levels": levels}}) or None


@cached(ttl=300)
async def fetch_abliteration_levels():
    # This fixed public endpoint receives no user/provider credentials.
    try:
        async with (
            aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10), trust_env=True
            ) as session,
            session.get(ABLITERATION_THINKING_URL) as response,
        ):
            response.raise_for_status()
            document = await response.text()
        return {
            model_id: levels
            for model_id in ABLITERATION_LARGE_MODELS
            if (levels := parse_abliteration_levels(document, model_id)) is not None
        }
    except (aiohttp.ClientError, TimeoutError):
        log.warning("Unable to load Abliteration reasoning capabilities")
        return {}


async def supplement_reasoning_capabilities(models):
    targets = []
    for model in models:
        if advertised_levels(model) is not None:
            continue
        # Identity must come from the upstream catalog, never a local alias name.
        identity = next(
            (
                model.get(k)
                for k in ("id", "name", "display_name")
                if model.get(k) in ABLITERATION_LARGE_MODELS
            ),
            None,
        )
        if identity:
            targets.append((model, identity))
    if not targets:
        return
    capabilities = await fetch_abliteration_levels()
    for model, identity in targets:
        if levels := capabilities.get(identity):
            thinking = model.get("thinking")
            model["thinking"] = {
                **(thinking if isinstance(thinking, dict) else {}),
                "levels": levels,
            }
            model["reasoning_capabilities_source"] = ABLITERATION_THINKING_URL
        else:
            log.warning("No verified reasoning levels for upstream model %s", identity)
