# Contributing

This repository follows a simple release and branching model so all TSuite nodes stay consistent.

## Branch strategy

- `main`: production-ready code only. Every release tag comes from this branch.
- `dev`: integration branch for upcoming work.
- `release/vX.Y.Z`: stabilization branch for a specific release.
- `hotfix/vX.Y.Z`: urgent production fixes branched from `main`.
- `feat/<short-name>`, `fix/<short-name>`, `chore/<short-name>`: short-lived branches, usually branched from `dev`.

## Versioning and tags

- `VERSION` is the source of truth and must use SemVer: `X.Y.Z`.
- On push to `main`, if `VERSION` changed, GitHub Actions creates tag `vX.Y.Z` automatically.
- Existing tags are never overwritten by automation.

## Standard release flow

1. Branch from `dev` (`feat/*`, `fix/*`, `chore/*`).
2. Open PR into `dev`.
3. Create `release/vX.Y.Z` from `dev` and set `VERSION` to `X.Y.Z`.
4. Merge `release/vX.Y.Z` into `main`.
5. Workflow creates tag `vX.Y.Z` from `main` automatically.
6. Merge `main` back into `dev`.

## Recommended GitHub protections

Set these in repository settings:

- Protect `main` and `dev`.
- Require pull requests before merge.
- Require status checks to pass.
- Restrict direct pushes to protected branches.
