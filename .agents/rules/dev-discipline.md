# ALERT Project — Development Discipline Rules

These rules are **always active** for the ALERT project. They enforce incremental
development suitable for clean git history and avoiding output token limit issues.

## Incremental Commit Rule

Every response that writes or modifies code must be scoped to a **single atomic commit**.
Do not implement more than one module, one feature, or one logical unit per response.

- Maximum ~150 lines of new logic per response
- Maximum 1–2 files created or modified per response
- All written functions must be complete and working — no stubs

## Mandatory Response Format

Every coding response must include, in order:

1. **Scope statement** — what this increment builds, what it depends on, what it unlocks
2. **Implementation** — the actual code
3. **Git commit message** — Conventional Commits format (feat/fix/chore/test/docs)
4. **Next steps** — 2–3 possible follow-up increments

## Conventional Commits Format

All suggested commit messages must follow:
```
<type>(<scope>): <short description>

<optional body>
```

## Git Operations — STRICTLY FORBIDDEN

**Never run any git command that modifies repository state.** The user handles all
version control operations themselves. This is an absolute, non-negotiable rule.

Forbidden commands (do NOT run under any circumstance):
- `git commit` — forbidden
- `git push` — forbidden
- `git add` — forbidden
- `git merge` — forbidden
- `git rebase` — forbidden
- `git tag` — forbidden
- `git stash` — forbidden

You MAY run read-only git commands when genuinely needed for context:
- `git status`, `git log`, `git diff`, `git branch` — allowed (read-only)

Only provide the commit message as **text to copy** — never execute it.

## Do Not

- Do not scaffold the entire project in one shot
- Do not leave placeholder/stub functions
- Do not skip the commit message at the end of a response
- Do not implement multiple modules in one response even if asked

## When a Request is Too Large

Break it into numbered increments, list them, then implement only the first one
(or ask the user which to start with).
