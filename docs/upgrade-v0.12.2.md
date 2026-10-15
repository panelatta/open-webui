# Open WebUI v0.12.2 local upgrade

The release is merged into the existing fork history, starting from `6a807952c`.
The official tag is `6a18585ff4c2e5f6805f682222cfbc46dc669ef8`.

## Local adaptations

- Memory vector reindexing uses the upstream batched call, routed through
  `embed_memory_content` and `MEMORY_EMBEDDING_FUNCTION`. This retains batching,
  provider concurrency limits and local memory prompt overrides. The generic
  `EMBEDDING_FUNCTION` remains disabled. The regression checks list input.
- Responses streaming retains contextual retries, background resume, status
  history and cleanup. The retry wrapper now propagates the upstream per-response
  usage result to tool-loop messages, preserving the new context accounting.
- Migration `a51d9012c822` joins upstream `0bd252b206fb` and local `62fbe170a903`.
  Existing migration files remain unchanged; both deployed histories are valid.
- Hosted attachment upload, memory extraction, account linking, security isolation,
  chat management, language switching and reasoning capability UI are retained.
- Of 113 locally customized files, upstream touched 29. The other 84 are unchanged
  except the memory reindex regression adapted for batched embeddings.

## Build and deployment

Build in the Lima VM using Node 22, 8 GiB RAM, existing 8 GiB swap and
`BUILD_MAX_OLD_SPACE_MB=6144 npm run build:safe`. Keep the existing safe build script.
Use the v0.12.2 runtime image with the complete locally compiled frontend and local
backend bind mounts. Change only the release tag and build hash in deployment
`.env`; preserve OAuth and security configuration.

Before cutover, snapshot databases, vector indexes, uploads and deployment
configuration. Validate the upgrade against a separate data copy, check SQLite
integrity and compare original foreign-key violations and record counts.
Run local backend/frontend regressions, memory and hosted-file smoke tests,
and browser upload checks. Confirm `/health`, `/api/version`, `/_app/version.json`
and SHA256 equality of the complete deployed frontend. Keep the pre-upgrade
source branch, image, frontend and data snapshot for rollback.

Detailed execution evidence is stored in the VM workspace under
`openwebui-upgrade-worktrees/validation-v0.12.2-20261015`.
