# SpectrumAI Copilot Pre-Flight Instructions

> **Self-Improvement Rule**: After every bot review cycle, update this file with any new
> patterns discovered. Add to the frequency table, quote the bot's exact words, and
> adjust the checklist. This file is the living rulebook — it only gets better over time.

> **Last Updated**: 2026-02-27 — Added PR lifecycle patterns from gh-resolve session (stale reviews, requirements pinning, logger levels)

---

## The Bot's 4 Review Principles

The `atos-spectrum-developer[bot]` enforces these principles **by name** in its reviews.
Every PR must pass all four or it gets `CHANGES_REQUESTED`.

| Principle | What the Bot Checks | Frequency |
|---|---|---|
| **Surgical Changes** | "Touch only what you must" — no unrelated reformatting, no cosmetic changes, no scope creep. Every file in the diff must be justified by the PR's stated goal. | **#1 reason for rejection** (42% of PRs) |
| **Think Before Coding** | PR title/description must accurately describe what the diff actually changes. Bot cross-checks description against the file list. | 32% of PRs |
| **Goal-Driven Execution** | New code must ship with tests. CI must be verified working. Stated goals must be fully delivered. | Referenced in 16% of PRs |
| **Simplicity First** | Pin dependencies, prefer simple solutions, no over-engineering. | Referenced in 5% of PRs |

---

## Pre-Flight Checklist

Run through EVERY item before pushing. If any item fails, fix it before committing.

### 1. SCOPE — The #1 Killer (42% of rejections)

- [ ] **Every file in `git diff --stat` is directly related to the PR's stated purpose**
- [ ] No formatting-only changes (whitespace, quote style, dash characters)
- [ ] No cosmetic file renames or reorganizations unless that IS the PR purpose
- [ ] No unrelated CI workflow changes bundled with feature work
- [ ] No "while I'm here" fixes — those go in separate PRs
- [ ] Run: `git diff origin/main --stat` and verify every listed file is justified

> Bot quote: "PR bundles 5+ unrelated concerns (Langfuse sync, docs overhaul, setup
> script replacement, submodule, CI housekeeping) violating the Surgical Changes guideline."

### 2. PR DESCRIPTION — Must Match the Diff (32% of rejections)

- [ ] PR title accurately reflects the actual changes
- [ ] PR body lists every file category being changed
- [ ] No "aspirational" descriptions of features not yet implemented
- [ ] If PR delivers more than described, update the description FIRST
- [ ] Run: compare `git diff --stat origin/main` output against PR description

> Bot quote: "PR delivers 25 new docs/skills/ files (~3,000 lines) but the description
> only mentions link fixes, outdated content, and filename normalisation."

### 3. HARDCODED VALUES & SECRETS (32% of rejections)

- [ ] No hardcoded secrets, API keys, HMAC keys, or tokens in code
- [ ] No hardcoded fallback package lists that can drift from pyproject.toml
- [ ] Dependencies pinned to specific versions or SHA (no floating `@main`)
- [ ] URLs use correct hostname (`atos.ghe.com` not `github.com`)
- [ ] Run: `grep -rn "secret\|password\|token\|api.key\|hmac" --include="*.py"` on changed files

> Bot quote: "The hard-coded fallback HMAC secret 'spectrumai-default-key' must be removed."
>
> Bot quote: "spectrumai-agent-framework @ git+https://... with no tag or SHA pin tracks
> main, meaning any breaking change will silently affect all repos."

### 4. CI & GITHUB ACTIONS (26% of rejections)

- [ ] **No `continue-on-error: true`** — use explicit output flags instead
- [ ] **Valid action versions** — `actions/checkout@v4`, `actions/setup-python@v5`. There is NO `@v6` for either.
- [ ] CodeQL job present unless default setup is API-verified: `gh api repos/{org}/{repo}/code-scanning/default-setup`
- [ ] No BOM characters in YAML files (check: `file .github/workflows/*.yml`)
- [ ] **CI pip installs use pinned versions or requirements files** — never `pip install pkg` with floating versions
- [ ] Token/PAT cleanup after git URL rewriting in CI (`if: always()` step)
- [ ] Silent failures (`|| true`) replaced with explicit `echo "::warning::..."` messages
- [ ] No cosmetic quote-style changes in workflow YAML
- [ ] CI checklist in PR body has all boxes checked ✅
- [ ] **Python version in CI matches pyproject.toml** — if `requires-python = ">=3.12"`, don't use `python-version: "3.11"`

> Bot quote: "continue-on-error: true has a silent-failure risk"
>
> Bot quote: "actions/setup-python@v6 is an invalid version. Please update to @v5."
>
> Bot quote: "pip install langfuse pyyaml uses floating versions in the CI job. Please
> pin them or use a requirements file to prevent silent supply-chain drift."

**Common CI Fix Pattern (found in 6+ repos on Feb 21):**
```yaml
# WRONG — @v6 does not exist for either action
uses: actions/checkout@v6
uses: actions/setup-python@v6

# CORRECT
uses: actions/checkout@v4
uses: actions/setup-python@v5
```

### 5. FORMATTING & COSMETICS (26% of rejections)

- [ ] No quote-style changes (single→double or vice versa) unless functionally required
- [ ] No whitespace-only reformatting of existing code
- [ ] No em-dash, box-drawing, or Unicode character substitutions
- [ ] No line-wrapping changes on existing code
- [ ] **No comment separator width changes** — `# ── Tools ───...` lines must keep their original width
- [ ] If a formatter is involved, only format files YOU actually changed functionally

> Bot quote: "Four multi-line expressions were reformatted to single lines — this is
> unrelated to CADE-292 and violates the 'touch only what you must' principle."
>
> Bot quote: "6 unnecessary quote-style changes (single → double quotes) ...
> cosmetic and unrelated to fixing action versions — please revert them."
>
> Lesson (Feb 21): Even 1-character width changes to `# ── Section ───` separator
> comments get flagged. Always compare: `git show origin/main:<file> | Select-String "# ──"`

### 6. TESTS & DOCUMENTATION (21% of rejections)

- [ ] Every new module/function has a corresponding test file
- [ ] New docs/examples are indexed in the relevant README
- [ ] **No dead cross-references** to files that don't exist in the repo
- [ ] Draft/WIP documentation is not merged to main
- [ ] **Check default branch name** — some repos use `master`, not `main`. Run: `gh api repos/GLB-Spectrum-AI/{repo} --jq .default_branch`
- [ ] **Python version consistent across all docs** — GETTING_STARTED.md, README, pyproject.toml, setup scripts must all agree
- [ ] **No cross-document contradictions** — if two docs describe the same feature, they must agree

> Bot quote: "210 lines of new logic ship with zero unit tests, and the PR itself
> acknowledges that CI hasn't been verified on main yet."
>
> Bot quote: "A developer landing on the repo root will not discover the new skill.
> Please add an entry for the new skill alongside existing entries."
>
> Bot quote: "Prerequisites says 'Python 3.11+ installed' but pyproject.toml and both
> setup scripts now require >=3.12. New franchisees on 3.11 will read the guide, pass
> the prose check, and then hit errors at install time."
>
> Lesson (Feb 21): langfuse-docs formerly used `master` not `main`. Content migrated to
> spectrumai-langfuse (which uses `main`). Always verify the default branch.

### 7. ERROR HANDLING & DEPENDENCIES

- [ ] Required env vars validated at startup (fail-fast, not silent fallback)
- [ ] Dependencies in pyproject.toml or requirements.txt, not inline in CI
- [ ] Import statements at module level (not inline inside functions/except blocks)

> Bot quote: "LANGFUSE_HOST is not validated alongside the other required env vars,
> so a missing host silently falls back to cloud.langfuse.com."

### 8. GIT HYGIENE & MERGE STATE

- [ ] Branch is rebased on latest main (no merge conflicts)
- [ ] No `.code-workspace` files tracked (add to .gitignore)
- [ ] No alembic migration files modified in-place (create new ones)
- [ ] No .gitmodules changes without documentation
- [ ] **Clean commit history** — squash noisy intermediate commits before review. Bots analyze commit messages.
- [ ] Run: `gh pr view {N} --json mergeable` confirms `MERGEABLE`

> Bot quote: "Alembic migrations are immutable once applied. Please revert those files
> to their committed state on main and create a new migration file."
>
> Lesson (Feb 21): starter #11 had 35 commits including intermediate adds/reverts that
> confused the bot into seeing changes that were net-zero in the final diff. Squashing
> to 1 clean commit resolved it.

---

## Bot Review Pattern Frequency Table

Maintained after each review cycle. Update counts when new reviews arrive.

| Pattern | Count | % of 22 PRs | Trend |
|---|---|---|---|
| Scope creep / out-of-scope files | 8 | 36% | ↓ fixing |
| **Invalid GitHub Action versions (@v6)** | **7** | **32%** | **NEW — #2 issue** |
| PR description mismatch | 6 | 27% | — |
| Hardcoded values / secrets / unpinned refs | 6 | 27% | — |
| Formatting-only / cosmetic changes | 6 | 27% | ↑ +1 |
| CI workflow misconfiguration | 5 | 23% | — |
| **Floating pip installs in CI** | **5** | **23%** | **NEW** |
| Dead cross-references / broken links | 5 | 23% | ↑ +2 |
| Missing / incomplete documentation | 4 | 18% | — |
| Missing tests for new code | 3 | 14% | — |
| CodeQL removal without evidence | 3 | 14% | — |
| Missing error handling / silent failures | 3 | 14% | — |
| **Python version inconsistency across docs** | **2** | **9%** | **NEW** |
| **Default branch name mismatch (main vs master)** | **2** | **9%** | **NEW** |
| **Cross-document contradictions** | **2** | **9%** | **NEW** |
| Dependency management (unpinned/floating) | 2 | 9% | — |
| Merge conflicts | 2 | 9% | — |
| Breaking backward compatibility | 2 | 9% | — |
| **Noisy commit history confusing bot** | **1** | **5%** | **NEW** |
| Encoding / BOM issues | 1 | 5% | — |
| Shell injection in setup scripts | 1 | 5% | — |
| Import style (inline vs module-level) | 1 | 5% | — |
| Immutable artifact modification | 1 | 5% | — |
| Token leakage in CI | 1 | 5% | — |

---

## Quick Commands

```bash
# Check scope — every file must be justified
git diff origin/main --stat

# Check for secrets/hardcoded values
grep -rn "secret\|password\|token\|api.key\|hmac\|default-key" --include="*.py" src/ scripts/

# Check for BOM in YAML
file .github/workflows/*.yml

# Check mergeable state
gh pr view {N} --json mergeable

# Check CodeQL default setup
gh api repos/GLB-Spectrum-AI/{repo}/code-scanning/default-setup --hostname atos.ghe.com

# Verify action versions are valid (MUST NOT have @v6 for checkout or setup-python)
grep -rn "uses:.*@v" .github/workflows/*.yml

# Quick check for the most common invalid version
grep -rn "@v6" .github/workflows/*.yml

# Check for continue-on-error
grep -rn "continue-on-error" .github/workflows/*.yml

# Check for floating pip installs in CI
grep -rn "pip install " .github/workflows/*.yml | grep -v "requirements\|==" | grep -v "\-e \."

# Check Python version consistency across all docs
grep -rn "Python 3\." GETTING_STARTED.md README.md pyproject.toml setup.sh setup.ps1 2>/dev/null

# Check default branch name (some repos use master)
gh api repos/GLB-Spectrum-AI/{repo} --jq .default_branch
```

---

## Subagent & PowerShell Rules

When delegating multi-repo work to subagents, these rules prevent common failures:

### 1. NEVER use loop variables in gh/git commands inside subagent prompts

Subagents run PowerShell commands in VS Code terminals. If a prompt contains
`$repo` or `$pr` inside double-quoted strings, PowerShell may fail to expand
them — producing literal paths like `spectrumai-agent-framework\$repo` which
trigger VS Code's "File write operations detected" approval dialog, blocking
the session.

**WRONG — variable may not expand, triggers file-write dialog:**
```powershell
foreach ($repo in $repos) {
  gh api repos/GLB-Spectrum-AI/$repo/pulls --hostname atos.ghe.com
}
```

**RIGHT — enumerate repos explicitly in the subagent prompt:**
```
For each of these repos, run the command with the repo name hardcoded:
1. gh api repos/GLB-Spectrum-AI/spectrumai-agent-starter/pulls ...
2. gh api repos/GLB-Spectrum-AI/spectrumai-agent-docs/pulls ...
3. gh api repos/GLB-Spectrum-AI/spectrumai-franchisee-project-starter/pulls ...
```

### 2. Prefer one command per repo over PowerShell loops

Subagent terminal sessions are shared. A failing loop iteration can corrupt
the session state for subsequent iterations. Instead, tell the subagent to
run separate explicit commands for each repo/PR.

### 3. Always set `$env:GH_HOST` before every `gh` command

Subagent terminals do not inherit environment variables from prior sessions.
Every `gh` command must be preceded by `$env:GH_HOST = "atos.ghe.com"`.

### 4. Never use `cd` with unexpanded variables

If a subagent uses `cd "C:\...\$repo"` and `$repo` is empty or unexpanded,
it navigates to the wrong directory and subsequent git commands affect the
wrong repo. Always use hardcoded full paths.

---

## Bot Trigger Mechanism

To request a re-review after fixing issues:

1. Remove `atos-reviewed` label from the tracking issue in `GLB-Spectrum-AI/org-admin`
2. Dispatch the review workflow:
   ```
   gh api -X POST "repos/GLB-Spectrum-AI/org-admin/actions/workflows/5199142/dispatches" -f ref=main --hostname atos.ghe.com
   ```
3. Each dispatch picks up the oldest unreviewed tracking issue

---

## Repository Map

| Repo | Local Path | Type |
|---|---|---|
| spectrumai-agent-framework | spectrumai-agent-framework | Core SDK |
| spectrumai-agent-templates | spectrumai-agent-templates | Agent scaffolding |
| spectrumai-agent-examples | spectrumai-agent-examples | Reference implementations |
| spectrumai-agent-docs | spectrumai-agent-docs | MkDocs documentation |
| spectrumai-agent-platform | spectrumai-agent-platform | Platform CLI |
| spectrumai-agent-starter | spectrumai-agent-starter | Starter template |
| spectrumai-franchisee-project-starter | spectrumai-franchisee-project-starter | Franchisee template |
| spectrumai-langfuse | spectrumai-langfuse | Langfuse platform (observability) |
| spectrumai-n8n-docs | spectrumai-n8n-docs | n8n workflow docs |
| spectrumai-de-project | spectrumai-de-project | DE project |
| spectrumai-de-agents | spectrumai-de-agents | DE agents |

All repos under `GLB-Spectrum-AI` org on `atos.ghe.com`.

---

## Repos with Non-Standard Default Branch

All repos use `main` as default branch.

Always verify before referencing branch names in docs or CI triggers.

---

## Valid GitHub Action Versions (as of Feb 2026)

| Action | Valid Latest | Invalid |
|---|---|---|
| `actions/checkout` | `@v4` | `@v5`, `@v6` do not exist |
| `actions/setup-python` | `@v5` | `@v6` does not exist |
| `actions/upload-artifact` | `@v4` | — |
| `actions/download-artifact` | `@v4` | — |
| `github/codeql-action/*` | `@v4` | — |
| `L-Space/git-release` | `@v1` | — |

---

## Changelog

| Date | Change | Evidence |
|---|---|---|
| 2026-02-22 | Added Subagent & PowerShell Rules section | Subagent loop variables ($repo) produced literal paths, triggering VS Code file-write dialogs and blocking sessions |
| 2026-02-22 | Updated with 6 new patterns from fixing 18 PRs across 10 repos | Invalid @v6 actions (#2 issue), floating CI deps, Python version mismatches, default branch confusion, separator width cosmetics, noisy commit histories |
| 2026-02-21 | Initial creation from analysis of 19 CHANGES_REQUESTED reviews across 10 repos | ~53 individual issues catalogued, frequency table established |
| 2026-02-27 | Added: stale CHANGES_REQUESTED pattern — latest COMMENTED review saying "Approve" is the true signal; fix by updating PR description to match actual diff | PR #83 |
| 2026-02-27 | Added: requirements.txt must cap ALL packages with upper-bound (<X.0.0) — bot flags any missing caps as supply-chain drift risk | PR #90 infisicalsdk |
| 2026-02-27 | Added: logger level for operational warnings — use INFO not DEBUG for "service unavailable, using fallback" messages so they survive default WARNING root level in production | PR #91 rate_limit.py |
| 2026-02-27 | Added: org audit workflow 5186267 creates tracking issues; review bot 5199142 processes oldest unreviewed one per dispatch; close/reopen PRs to re-trigger triage agent labeling | PR triage session |
