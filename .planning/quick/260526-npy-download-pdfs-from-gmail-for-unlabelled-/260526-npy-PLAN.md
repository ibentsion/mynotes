---
phase: quick-260526-npy
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - pyproject.toml
  - scripts/download_gmail_pdfs.py
autonomous: false
requirements: [QUICK-NPY]
user_setup:
  - service: google-oauth
    why: "First-run Gmail OAuth consent to download PDF attachments"
    env_vars: []
    dashboard_config:
      - task: "Authorize the app in the browser when prompted on first run"
        location: "Browser OAuth consent screen (project: mynotes-hebrew-pdfs)"

must_haves:
  truths:
    - "Running the script downloads new PDF attachments from Gmail into data/pdfs/"
    - "PDFs already present in data/pdfs/ are skipped, not re-downloaded"
    - "Script prints how many attachments were found, skipped, and downloaded"
    - "OAuth token is cached outside the repo so subsequent runs need no browser"
  artifacts:
    - path: "scripts/download_gmail_pdfs.py"
      provides: "Standalone Gmail PDF downloader CLI"
      min_lines: 50
    - path: "pyproject.toml"
      provides: "Google API client dependencies"
      contains: "google-api-python-client"
  key_links:
    - from: "scripts/download_gmail_pdfs.py"
      to: "~/.config/gws/client_secret.json"
      via: "InstalledAppFlow.from_client_secrets_file"
      pattern: "client_secret"
    - from: "scripts/download_gmail_pdfs.py"
      to: "data/pdfs/"
      via: "filename existence check before write"
      pattern: "data/pdfs"
---

<objective>
Add a standalone script that downloads PDF attachments from Gmail into `data/pdfs/` to grow the pool of unlabelled OCR source data.

Purpose: The OCR pipeline needs more scanned note PDFs as raw input. They arrive by email; this automates fetching them without manual download.
Output: `scripts/download_gmail_pdfs.py` plus pinned Google API dependencies in `pyproject.toml`.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@./CLAUDE.md

<facts>
- OAuth client secret exists at: ~/.config/gws/client_secret.json (type: installed, project: mynotes-hebrew-pdfs)
- `data/` is gitignored (privacy-sensitive). data/pdfs/ already holds ~10 PDFs named like "רשימות אימון 01-11-2026 .pdf".
- Google API libs are NOT installed yet. Pin exactly per project convention (== not >=):
  - google-api-python-client==2.169.0
  - google-auth-oauthlib==1.2.1
  - google-auth-httplib2==0.2.0
- Token cache must live OUTSIDE the repo: ~/.config/gws/gmail_token.json
- Run directly: `uv run python scripts/download_gmail_pdfs.py`
- No reusable entry point / packaging needed — this is a standalone utility script.
</facts>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add Google API dependencies</name>
  <files>pyproject.toml</files>
  <action>
    Add to the `dependencies` list in pyproject.toml, with exact pinned versions:
    - "google-api-python-client==2.169.0"
    - "google-auth-oauthlib==1.2.1"
    - "google-auth-httplib2==0.2.0"
    Then run `uv sync` to install. Do not modify the dev group or override-dependencies.
  </action>
  <verify>
    <automated>uv sync && uv run python -c "import googleapiclient.discovery, google_auth_oauthlib.flow, google_auth_httplib2"</automated>
  </verify>
  <done>The three Google packages are pinned in pyproject.toml and import cleanly in the venv.</done>
</task>

<task type="auto">
  <name>Task 2: Write the Gmail PDF downloader script</name>
  <files>scripts/download_gmail_pdfs.py</files>
  <action>
    Create a standalone script (no packaging, no entry point) that:
    - Defines constants: CLIENT_SECRET = Path.home()/".config/gws/client_secret.json",
      TOKEN_PATH = Path.home()/".config/gws/gmail_token.json",
      DEST_DIR = Path("data/pdfs"), SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"].
    - Auth: load cached Credentials from TOKEN_PATH if present; if missing/invalid and refreshable, refresh;
      otherwise run InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET, SCOPES).run_local_server(port=0).
      Persist credentials back to TOKEN_PATH (mode 0o600 via TOKEN_PATH.write_text + chmod). Fail fast with a
      clear message if CLIENT_SECRET is missing.
    - Build the Gmail service: build("gmail", "v1", credentials=creds).
    - Search: messages.list with q="has:attachment filename:pdf", paginating via nextPageToken until exhausted.
    - For each message, fetch full message, walk payload parts, and for each part whose filename ends in .pdf:
      decode body via attachments.get when body has attachmentId, base64.urlsafe_b64decode the data.
      Skip (count as skipped) if DEST_DIR/<filename> already exists. Otherwise write bytes to DEST_DIR/<filename>
      (count as downloaded). DEST_DIR.mkdir(parents=True, exist_ok=True) before writing.
    - Track and print a final summary: "<found> PDF attachments found, <skipped> skipped (already present),
      <downloaded> downloaded into data/pdfs/".
    - Keep each function under 100 lines / complexity <=8 per CLAUDE.md; no docstrings on private helpers,
      no inline comments on obvious logic, absolute imports only. Fail fast with actionable errors (which message,
      which filename) rather than swallowing exceptions.
    - `if __name__ == "__main__": main()`.
  </action>
  <verify>
    <automated>uv run python -c "import ast; ast.parse(open('scripts/download_gmail_pdfs.py').read())" && uv run ruff check scripts/download_gmail_pdfs.py && uv run ty check scripts/download_gmail_pdfs.py</automated>
  </verify>
  <done>Script parses, passes ruff and ty, and contains the auth + search + skip-existing + download + summary logic.</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <what-built>scripts/download_gmail_pdfs.py — downloads new Gmail PDF attachments into data/pdfs/, skipping files already present, with a printed found/skipped/downloaded summary.</what-built>
  <how-to-verify>
    1. Run: `uv run python scripts/download_gmail_pdfs.py`
    2. On first run, complete the OAuth consent in the browser that opens (project mynotes-hebrew-pdfs).
    3. Confirm the summary line prints found/skipped/downloaded counts.
    4. Confirm `ls data/pdfs/` shows the existing 10 PDFs untouched plus any newly downloaded ones.
    5. Re-run the script; confirm everything is now reported as skipped (0 downloaded) and a token file exists at ~/.config/gws/gmail_token.json so no browser opens.
  </how-to-verify>
  <resume-signal>Type "approved" or describe issues</resume-signal>
</task>

</tasks>

<verification>
- `uv sync` installs the three Google packages and they import.
- Script passes ast.parse, ruff, and ty.
- Manual run downloads new PDFs, skips existing ones, prints a summary, and caches a token outside the repo.
</verification>

<success_criteria>
- Running the script populates data/pdfs/ with new Gmail PDF attachments without re-downloading existing files.
- A clear found/skipped/downloaded summary is printed.
- OAuth token cached at ~/.config/gws/gmail_token.json (outside the gitignored repo data path) for credential reuse.
</success_criteria>

<output>
After completion, create `.planning/quick/260526-npy-download-pdfs-from-gmail-for-unlabelled-/260526-npy-SUMMARY.md`
</output>
