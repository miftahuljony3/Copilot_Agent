# Repository guidance for Copilot

## General rules

- Inspect the repository before recommending or making changes.
- Treat the checked-in files as the source of truth; do not invent project structure, dependencies, commands, or behavior.
- Separate verified observations from assumptions and recommendations.
- If essential information is missing, say what is missing and ask a focused question.
- Keep changes scoped to the user’s request.
- Prefer small, reviewable changes over broad rewrites.
- Do not expose secrets. Do not copy secret values into logs, documentation, tests, or examples.

## Before changing files

1. Identify the relevant files and existing conventions.
2. Read nearby code and documentation.
3. Check the project’s manifests and existing scripts for the correct build, test, and lint commands.
4. If a requested change could cause data loss, security exposure, or an irreversible action, explain the risk and request confirmation before proceeding.

## When changing files

- Follow existing formatting and naming conventions.
- Avoid adding dependencies unless they are needed; explain why when adding one.
- Update relevant documentation when behavior, setup, or configuration changes.
- Add or update tests when the project has a suitable test setup.
- Do not claim tests passed unless they were actually run successfully.

## When reporting work

Summarize:
- What changed.
- Which files changed.
- What checks were run and their results.
- Any remaining risks, assumptions, or follow-up work.
