# Open WebUI v0.12.0 fork integration

This merge retains the fork history through `61f932f72` and merges the official
v0.12.0 release. Deployment must build the frontend locally with Node 22,
`BUILD_MAX_OLD_SPACE_MB=4096 npm run build:safe`, and 8 GiB swap.

## Compatibility adaptations

- Keep upstream chat error publication, approved-tool handling, model controls,
  and cleanup structure; insert the existing reasoning placeholder into that flow.
- Keep generation-status deduplication while also updating the new chat-message
  status storage. Preserve chat management and folder move timestamps.
- Use upstream shared provider header/OAuth helpers. Retain hosted file uploads,
  expired-file retries, generation status, reasoning selection, and Responses resume.
  Preserve upstream Retry-After headers for both SSE and regular errors.
- Keep the original request's tools snapshot before model defaults are applied.
  A later upstream snapshot would incorrectly suppress builtins for hosted search presets.
- Use upstream file-context assembly from message attachments before metadata is
  stripped, replacing the old database re-read path. Retain attachment detection
  for contextual retries and suppress local file RAG for OpenAI models.
- Use upstream centralized tool execution and parameter parsing; retain stream
  retry/resume and synchronize output offsets with the upstream incremental stream.
- Adapt memory-only embedding policy to the new retrieval config dictionary and
  Config.upsert flow. Generic embedding remains disabled; memory retains its own
  callable. Web search and URL extraction do not create vector collections.
- Retain channel upload process=false with upstream image preparation and upload
  queue handling; chat uploads remain process=false with hosted file routing.
- Integrate custom reasoning settings into the new reactive model editor and
  retain hidden-capability filtering, language controls, and folder layout.

## Verification before deployment

The v0.12.0 runtime passes 214 backend regression cases, including Responses
streaming/retry/resume/status, hosted payloads, memory-only policy, reasoning,
configured tools, cache behavior and chat management. The five local frontend
suites pass 19 tests. Undefined-name checks and Python compilation pass.

Deployment verification must additionally confirm /health, /api/version,
/_app/version.json, build checksums, database/vector integrity, real hosted
attachment reading, and memory embedding write/query using the configured cloud
provider. Preserve a consistent backup before the new migrations are applied.
