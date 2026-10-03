# GLM task checklist and hosted search

The GLM preset previously exposed task and time functions but no web search.
A checklist records progress; it does not schedule work after a completed model
response. The affected response executed its functions successfully, then ended
with an announcement instead of delivering the requested research. There was no
saved stream error or unhandled function call.

Enable provider-hosted search on the existing preset using
`scripts/configure-glm-hosted-search.py`. Run it inside the Open WebUI container
with `--backup /path/to/new-backup.json --apply`; without `--apply` it only backs up
the model. The backup path must not already exist. This requires the container's
database and signing key and uses the admin model API to update configuration.
Keep backups private and outside Git. Restore through the model update API if
needed.

The script adds `custom_params.tools = [{"type": "web_search"}]` (preserving any
existing tools), defaults `tool_choice` to `auto`, marks the web search capability,
and appends guidance to complete work after creating tasks and report unavailable
tools honestly. It preserves reasoning levels, the model identity, access grants,
task tools, and other capabilities. Search is available to the model automatically;
this does not enable Open WebUI's global local-search setting or its search button.
Memory, local search tools, and file capabilities retain their previous settings.

`process_chat_payload` now snapshots caller-supplied tools before expanding model
parameters. Previously `custom_params.tools` was mistaken for an explicit caller
override, silently suppressing all builtins. Preset and inlet-filter tools now
merge with resolved builtin tools. Explicit top-level `tools`, including `[]`,
still opt out of server-side resolution. API callers without a browser session
still receive no hidden builtin tools.

Regression coverage exercises the real payload processor with preset search,
browser and API callers, and explicit empty/nonempty tool lists. Live verification
should use a disposable saved chat asking for an IANA example-domain search with
`create_tasks`, `update_task`, and a substantive final answer. Confirm completed
hosted `web_search_call` items, successful task outputs, and the final answer;
delete only that disposable chat afterwards. No original conversation should be
rewritten or regenerated automatically.

This changes tool availability and corrects tool-list handling; it cannot guarantee
that every future model response will finish every requested task. Do not add an
unbounded retry loop when a model finishes with pending checklist items.
