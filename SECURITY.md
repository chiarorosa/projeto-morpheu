# Security Policy

## Supported versions

Only the latest commit on `main` receives fixes.

## Reporting a vulnerability

Please **do not open a public issue**. Report it privately through
[GitHub's private vulnerability reporting](https://github.com/chiarorosa/projeto-morpheu/security/advisories/new),
with steps to reproduce and the impact you expect. You should get an answer within a week.

## Scope notes

- The API binds to `127.0.0.1` and has **no authentication** by design. Exposing it to a network
  (port forwarding, `--host 0.0.0.0` patches, tunnels) lets anyone use your GPU and read your voice library.
- Model weights are downloaded from Hugging Face on first use; review the model repositories you enable.
