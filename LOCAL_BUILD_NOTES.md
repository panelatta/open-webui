# Local Build Notes

## Frontend Build On Low-Resource Hosts

Do not run unconstrained `vite build` directly on this Raspberry Pi class host.
The previous incident showed that a normal frontend build can saturate CPU and
memory badly enough to destabilize the machine.

Preferred order:

1. Build the frontend on a stronger machine.
2. If this host must build it, use `npm run build:safe`.

`npm run build:safe` does two things:

- lowers CPU and IO priority with `nice` and `ionice` when available
- limits Node heap with `NODE_OPTIONS=--max-old-space-size=768` by default

Optional overrides:

- `BUILD_MAX_OLD_SPACE_MB=512 npm run build:safe`
- `SAFE_BUILD_SKIP_PYODIDE_FETCH=1 npm run build:safe`

Do not selectively patch compiled frontend files in runtime deployment.
Frontend changes must come from source and be deployed as one coherent build.
