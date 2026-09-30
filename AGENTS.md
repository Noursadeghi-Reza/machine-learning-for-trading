# AGENTS.md

## Project Purpose

This fork is used to study and extend Stefan Jansen's
Machine Learning for Trading repository while preserving
the upstream project as the technical baseline.

## Upstream Relationship

- `upstream/main` is the canonical upstream source.
- `origin/main` should stay close to `upstream/main`.
- Personal research and workflow changes must be made on branches.
- Do not make experimental changes directly on `main`.

## Research Integrity

For financial ML work:

- Preserve point-in-time correctness.
- Do not introduce lookahead bias or target leakage.
- Treat future-return columns carefully and verify how they are used.
- Distinguish research hypotheses from validated findings.
- Do not claim an experiment succeeded without evidence.

## Change Discipline

Before editing:

1. Inspect the relevant code, config, tests, and documentation.
2. Make the smallest targeted change.
3. Do not perform broad refactors unless explicitly requested.
4. Do not modify unrelated notebooks or generated artifacts.

## Verification

For code changes:

- Run the smallest relevant test first.
- Run broader verification when the change affects shared infrastructure.
- Do not report a task as complete if tests or validation are still failing.

## Source of Truth

- Code, config, and tests: repository
- Implementation tasks: GitHub Issues
- Research notes and hypotheses: Obsidian
- Agent memory is context, not technical truth

## Human Approval

Do not perform destructive or high-impact actions without explicit approval, including:

- merging to `main`
- deleting branches or data
- changing credentials
- deployment
- live trading or broker actions

## Workflow

The current workflow is intentionally manual:

Obsidian
→ research idea
→ GitHub Issue
→ Cursor
→ implementation
→ tests
→ GitHub
→ result
→ Obsidian

Do not automate this workflow unless explicitly requested.