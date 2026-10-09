# Configurable chat reasoning levels

The input composer shows a small **Thinking effort** selector for supported connections. Each model has its own selection in the current chat, including custom models whose names do not identify their upstream models. Existing chats save choices in chat parameters; new chats start at the model's configured default. Changing a choice affects subsequent requests, including regeneration, and does not change an in-flight request.

In **Workspace → Models → Edit → Thinking effort levels**, choose **Custom** to add, delete, reorder and rename levels. Use a stable level ID, a request value and the language-code input to edit labels for any locale (`en-US`, `zh-CN`, `ja-JP`, etc.). Missing labels fall back through the current language family, English and the level ID. Choose **Inherit from base model** to use the nearest configured base model or connection defaults; **Disabled** hides the control.

**Model default** preserves existing advanced parameters. A custom default can be chosen separately. A deleted selection falls back to the current configured default when the browser prepares a new request. Requests from stale clients with unknown IDs are rejected instead of silently applying a different level. Choices are saved per chat, not as global account preferences.

## Request mappings

| Connection / protocol                                | Field                                                                            | Example value                                                 |
| ---------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| OpenAI Chat Completions                              | `reasoning_effort`                                                               | `low`, `medium`, `high`                                       |
| OpenAI Responses                                     | `reasoning.effort` (also accepts the automatically converted `reasoning_effort`) | Model-supported effort string                                 |
| Current OpenRouter Claude / OpenAI connection        | `reasoning.effort`                                                               | `low`, `medium`, `high`                                       |
| A gateway accepting Claude native effort fields      | `output_config.effort`                                                           | Model-supported effort string                                 |
| A gateway accepting Claude explicit thinking budgets | `thinking.budget_tokens`                                                         | `1024`, with additional binding `{"thinking.type":"enabled"}` |

Templates configure fields; they do not change the connection's API endpoint or add a native Anthropic transport. The current Claude models use OpenRouter. For other connections, verify the gateway's contract and the actual upstream model's supported values. Model-specific values such as `none`, `xhigh` or `max` can be added; support is not universal. Automatic choices first use the base model's provider metadata (`thinking.levels`, or `supported_reasoning_levels` effort entries), retaining provider order and exact request values. This applies to aliases as well as base models, and the model-list endpoint and request router use the same resolver. An explicit empty or malformed list hides automatic choices. Only when metadata is absent do recognized model names use the legacy conservative fallback; unknown models require explicit configuration. Saved custom configurations (including disabled controls) take precedence over provider metadata.

Do not add client-only modes just because a different app displays them. On 2026-10-08 the connected gateway advertises GPT-6 Astra as `low/medium/high/xhigh/max`; a real request with `max` completed with `reasoning.effort=max`, while `ultra` returned HTTP 400 (`invalid_value`). The resolver will expose `ultra` if a provider later advertises it, without a code change. GPT-5.5 currently advertises `low/medium/high/xhigh`. The saved GLM-5.3 Abliterated preset correctly retains `low/high/max` and its default `max`: [the provider's documentation](https://docs.abliteration.ai/capabilities/thinking) confirms these are Large V2's three distinct depths; other accepted names are aliases, not extra depths.

The mapping preview shows the dotted fields for every level. Additional bindings can set or remove reasoning fields; `null` removes a field. Allowed fields are `reasoning_effort`, `reasoning.effort`, `reasoning.max_tokens`, `reasoning.enabled`, `reasoning.exclude`, `thinking.type`, `thinking.budget_tokens`, and `output_config.effort`. Model IDs, messages, tools, headers and credentials cannot be changed through this mechanism. Budget presets require a compatible model, at least 1024 thinking tokens and a larger `max_tokens` setting in Advanced Params.

The server loads the saved configuration and resolves the selected ID after applying model defaults. A level overrides the reasoning fields it manages; other fields such as `reasoning.summary`, prompts and tool settings are retained. Competing effort aliases and stale additional bindings are removed. Frontend selection maps and IDs are never forwarded to the provider. Configuration is stored in model metadata; no database migration or existing model configuration rewrite is needed.

## Verification

Backend tests cover aliases, inheritance, disabling, cycles, invalid IDs/fields, budget validation, stale binding removal and captured outbound HTTP payloads for Responses, Chat Completions and OpenRouter. Frontend tests cover locale fallback, deleted choices and per-model request selection. Browser checks exercise desktop and narrow touch layouts, existing/new chats, refresh persistence, language changes, outgoing requests and model editor changes.

Official parameter references:

- https://developers.openai.com/api/docs/guides/reasoning
- https://docs.anthropic.com/en/docs/build-with-claude/effort

Deployment must use a complete local frontend build. The existing memory-only embedding policy, hosted chat file uploads, Responses retry/background resume and account language settings remain in place.

## Native clients (Open Relay)

Open Relay's Chat Settings → Advanced Params → Reasoning sends
`params.reasoning_effort`. The server now translates this explicit value when
no `reasoning_effort_level` is supplied. An explicit web preset (including the
empty model-default selection) retains priority. Omitting effort preserves
existing model defaults.

Available values come from the upstream catalog's `thinking.levels`,
`supported_reasoning_levels`, or OpenRouter's `reasoning.supported_efforts`.
Manual connection model lists remain allowlists but now fetch upstream metadata.
The compatibility path refreshes through the existing model-catalog cache on
each explicit effort request; it does not reuse app-state metadata indefinitely.
It never uses the legacy model-name low/medium/high list as proof of support.
Every advertised value is accepted, even when saved web presets contain only
a subset. Whitespace and case are normalized back to the provider's exact value.

The connected Abliteration gateway does not publish effort enums. For upstream
catalog entries identifying `abliterated-model-large-v2` or
`abliterated-model-large`, fetch
[the provider's capability document](https://docs.abliteration.ai/capabilities/thinking.md)
and parse its model-specific distinct modes, with a five-minute cache. This is
a documentation source, not a machine-readable model API. No effort enumeration
is hardcoded. Ambiguous/changed document syntax or fetch failure cannot manufacture
capabilities; explicit effort requests fail when a reasoning model has no
verified list. A machine-readable enum takes precedence, including an empty list.
The public documentation fetch receives no connection credentials.

Recognized values pass through unchanged. Named custom presets are also accepted
if their actual effort value is advertised. Otherwise GPT/o-series and Claude
choose advertised `xhigh`, then advertised `high`; GLM chooses advertised
`max`. If the requested fallback is unavailable, return a clear error listing
the actual options instead of sending an unsupported value. Other model families
require a recognized value. Non-reasoning models such as GPT-4o discard the
client effort override; they do not acquire reasoning from a previous chat's
defaults. Disabled controls stay disabled.

Protocol mapping uses the model's configured effort field/bindings when present,
otherwise `reasoning.effort` for OpenRouter and `reasoning_effort` for
OpenAI-compatible Chat Completions. Responses conversion produces
`reasoning.effort`. Model-owned default presets cannot override an explicit
client choice. Conflicting effort fields/stale bindings are cleared; summary,
tools and other unrelated fields remain. Budget-based custom presets still
require the explicit web preset protocol.

This change does not add a native picker to Open Relay or change its chat
parameter persistence: its unknown-key handling still does not round-trip the
web client's per-model selection map.
