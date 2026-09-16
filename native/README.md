# native/

First-party **production** operator implementations live here.

Gate stubs (`fx.*`) stay under `gate/fixture`. Third-party wrappers stay under `vendor/<name>/` (link libraries only; never copy algorithm repo sources). Gate-level `cloud.*` stubs were removed — ship production cloud ops here under `native/`.

## Operator doc contract (Studio「原理」)

Commercial / production ops **must** set `Usage.principle` (non-empty) at registration.
`Usage.notes` optional. Studio teachable ⇔ `principle` non-empty.
See workspace spec: `docs/superpowers/specs/2026-09-16-operator-doc-contract-design.md`.
