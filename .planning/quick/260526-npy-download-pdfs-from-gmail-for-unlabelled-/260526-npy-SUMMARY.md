---
phase: quick-260526-npy
plan: 01
subsystem: data-ingestion
tags: [gmail, oauth, pdf, data-collection]
dependency_graph:
  requires: []
  provides: [scripts/download_gmail_pdfs.py]
  affects: [data/pdfs/]
tech_stack:
  added:
    - google-api-python-client==2.169.0
    - google-auth-oauthlib==1.2.1
    - google-auth-httplib2==0.2.0
  patterns:
    - InstalledAppFlow OAuth with token caching at ~/.config/gws/gmail_token.json
key_files:
  created:
    - scripts/download_gmail_pdfs.py
  modified:
    - pyproject.toml
    - uv.lock
decisions:
  - Token cached outside repo at ~/.config/gws/gmail_token.json (mode 0o600) to avoid committing credentials
  - Standalone script with no entry point — not packaged, run via uv run python
  - Paginate messages.list with nextPageToken to handle inboxes with many PDF emails
metrics:
  duration_minutes: ~10
  completed_date: "2026-05-26"
  tasks_completed: 2
  files_changed: 3
---

# Quick 260526-npy: Gmail PDF Downloader Summary

**One-liner:** Standalone OAuth2 Gmail script that downloads PDF attachments into `data/pdfs/`, skipping existing files and printing a found/skipped/downloaded summary.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add Google API dependencies | 5982d59 | pyproject.toml, uv.lock |
| 2 | Write the Gmail PDF downloader script | 0fa83eb | scripts/download_gmail_pdfs.py |

## What Was Built

`scripts/download_gmail_pdfs.py` — a standalone CLI script that:

1. Authenticates via InstalledAppFlow (OAuth2) using the client secret at `~/.config/gws/client_secret.json`
2. Caches the token at `~/.config/gws/gmail_token.json` (0o600) for re-use without browser prompts
3. Searches Gmail for all messages with PDF attachments (`has:attachment filename:pdf`), paginating via `nextPageToken`
4. For each PDF attachment part: skips if `data/pdfs/<filename>` already exists, otherwise downloads and writes bytes
5. Prints: `<found> PDF attachments found, <skipped> skipped (already present), <downloaded> downloaded into data/pdfs/`

Three Google API packages pinned with exact versions in `pyproject.toml` and installed via `uv sync`.

## Verification

- `uv sync` resolved and installed all 17 new packages (including transitive deps like google-auth, oauthlib, httplib2)
- `import googleapiclient.discovery, google_auth_oauthlib.flow, google_auth_httplib2` — all import cleanly
- `ast.parse` — script parses
- `ruff check` — all checks passed (fixed 3 E501 line-length violations during development)
- `ty check` — all checks passed

## Stopped At (Checkpoint)

Task 3 is a `checkpoint:human-verify (blocking)` — the human must:
1. Run `uv run python scripts/download_gmail_pdfs.py`
2. Complete the OAuth consent in the browser (project: mynotes-hebrew-pdfs)
3. Confirm found/skipped/downloaded summary line prints
4. Confirm `data/pdfs/` shows existing PDFs untouched plus any new ones
5. Re-run to confirm idempotency (0 downloaded, token cached at `~/.config/gws/gmail_token.json`)

## Deviations from Plan

None — plan executed exactly as written, apart from fixing 3 ruff E501 violations (line-length) found during automated verification.

## Known Stubs

None. Script is complete end-to-end; the only unverified path is the live OAuth + Gmail API call, which requires the human-verify checkpoint.

## Self-Check: PASSED

- scripts/download_gmail_pdfs.py: FOUND
- pyproject.toml (contains google-api-python-client): FOUND
- Commit 5982d59: FOUND
- Commit 0fa83eb: FOUND
