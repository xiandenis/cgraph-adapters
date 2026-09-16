# native/

First-party **production** operator implementations live here.

Gate stubs stay under `gate/`. Third-party wrappers stay under `vendor/<name>/` (link libraries only; never copy algorithm repo sources).

## Operator doc contract (Studio「原理」)

Commercial / production ops **must** set `Usage.principle` (non-empty) at registration.
`Usage.notes` optional. Studio teachable ⇔ `principle` non-empty.
See workspace spec: `docs/superpowers/specs/2026-09-16-operator-doc-contract-design.md`.
