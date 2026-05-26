"""Download PDF attachments and training-log emails from Gmail."""
import base64
import re
import sys
from email.utils import parsedate_to_datetime
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

CLIENT_SECRET = Path.home() / ".config/gws/client_secret.json"
TOKEN_PATH = Path.home() / ".config/gws/gmail_token.json"
DEST_DIR = Path("data/pdfs")
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
QUERY = "has:attachment filename:pdf to:efi@elcoaching.co.il"
KEEP_PATTERNS = ("רשימות אימון", "רשימת אימון", "CamScanner")
TEXT_DIR = Path("data/text")
TEXT_SUBJECTS = ("רשימות אימון", "תובנות אימון")


def _get_credentials() -> Credentials:
    if not CLIENT_SECRET.exists():
        sys.exit(
            f"Client secret not found: {CLIENT_SECRET}\n"
            "Create an OAuth 'Desktop' app in GCP and download it there."
        )

    creds: Credentials | None = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET), SCOPES)
        creds = flow.run_local_server(port=0)

    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json())
    TOKEN_PATH.chmod(0o600)
    return creds


def _list_all_messages(service) -> list[str]:  # type: ignore[type-arg]
    msg_ids: list[str] = []
    response = service.users().messages().list(userId="me", q=QUERY).execute()
    while True:
        msg_ids.extend(m["id"] for m in response.get("messages", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
        response = service.users().messages().list(
            userId="me", q=QUERY, pageToken=page_token
        ).execute()
    return msg_ids


def _pdf_parts(payload: dict) -> list[dict]:  # type: ignore[type-arg]
    parts = payload.get("parts", [])
    if not parts:
        return [payload] if payload.get("filename", "").lower().endswith(".pdf") else []
    results: list[dict] = []  # type: ignore[type-arg]
    for part in parts:
        results.extend(_pdf_parts(part))
    return results


def _download_attachment(service, msg_id: str, part: dict) -> bytes:  # type: ignore[type-arg]
    body = part["body"]
    if "attachmentId" in body:
        att = service.users().messages().attachments().get(
            userId="me", messageId=msg_id, id=body["attachmentId"]
        ).execute()
        data = att["data"]
    else:
        data = body["data"]
    return base64.urlsafe_b64decode(data)



def _matches(filename: str) -> bool:
    return any(p in filename for p in KEEP_PATTERNS)


def cleanup(service) -> None:  # noqa: ARG001
    deleted = kept = 0
    for pdf in sorted(DEST_DIR.glob("*.pdf")):
        if not _matches(pdf.name):
            print(f"  Deleting: {pdf.name}")
            pdf.unlink()
            deleted += 1
        else:
            kept += 1

    print(f"{kept} kept, {deleted} deleted from data/pdfs/")


def download(service) -> None:
    print("Searching Gmail for PDF attachments sent to efi@elcoaching.co.il...")
    msg_ids = _list_all_messages(service)
    print(f"Found {len(msg_ids)} matching messages. Scanning...")
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    found = skipped = downloaded = 0

    for i, msg_id in enumerate(msg_ids, 1):
        print(f"  [{i}/{len(msg_ids)}]", end="\r", flush=True)
        message = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
        parts = _pdf_parts(message["payload"])
        for part in parts:
            filename = part.get("filename", "")
            if not filename.lower().endswith(".pdf"):
                continue
            if not _matches(filename):
                continue
            found += 1
            dest = DEST_DIR / filename
            if dest.exists():
                skipped += 1
                continue
            data = _download_attachment(service, msg_id, part)
            dest.write_bytes(data)
            downloaded += 1
            print(f"  Downloaded: {filename}")

    print(
        f"\n{found} PDF attachments found, {skipped} skipped (already present), "
        f"{downloaded} downloaded into data/pdfs/"
    )


def _header(headers: list[dict], name: str) -> str:  # type: ignore[type-arg]
    return next((h["value"] for h in headers if h["name"].lower() == name.lower()), "")


def _message_date(headers: list[dict]) -> str:  # type: ignore[type-arg]
    raw = _header(headers, "date")
    try:
        return parsedate_to_datetime(raw).strftime("%Y%m%d")
    except Exception:
        return "00000000"


def _safe_subject(subject: str) -> str:
    return re.sub(r'[<>:"/\\|?*\n\r]', "", subject).strip()


def _text_body(payload: dict) -> str:  # type: ignore[type-arg]
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data", "")
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace") if data else ""
    for part in payload.get("parts", []):
        result = _text_body(part)
        if result:
            return result
    return ""


def download_text(service) -> None:  # type: ignore[type-arg]
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    saved = skipped = 0

    for subject_filter in TEXT_SUBJECTS:
        query = f'to:efi@elcoaching.co.il subject:"{subject_filter}"'
        print(f'Searching: {query}')
        response = service.users().messages().list(userId="me", q=query).execute()
        msg_ids: list[str] = []
        while True:
            msg_ids.extend(m["id"] for m in response.get("messages", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break
            response = service.users().messages().list(userId="me", q=query, pageToken=page_token).execute()

        print(f"  {len(msg_ids)} messages found. Downloading...")
        for i, msg_id in enumerate(msg_ids, 1):
            print(f"  [{i}/{len(msg_ids)}]", end="\r", flush=True)
            msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
            headers = msg["payload"].get("headers", [])
            date_str = _message_date(headers)
            subject = _safe_subject(_header(headers, "subject"))
            dest = TEXT_DIR / f"{date_str}_{subject}.txt"
            if dest.exists():
                skipped += 1
                continue
            body = _text_body(msg["payload"])
            if len(body.encode()) < 100:
                skipped += 1
                continue
            dest.write_text(body, encoding="utf-8")
            saved += 1
            print(f"  Saved: {dest.name}")

    print(f"\n{saved} saved, {skipped} skipped (already present) into data/text/")


def fetch_one(service, target: str) -> None:
    """Download a single PDF by exact filename, overwriting if present."""
    query = f'has:attachment filename:"{target}"'
    response = service.users().messages().list(userId="me", q=query).execute()
    msg_ids = [m["id"] for m in response.get("messages", [])]
    if not msg_ids:
        sys.exit(f"No Gmail message found with attachment: {target}")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for msg_id in msg_ids:
        message = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
        for part in _pdf_parts(message["payload"]):
            if part.get("filename") == target:
                data = _download_attachment(service, msg_id, part)
                dest = DEST_DIR / target
                dest.write_bytes(data)
                print(f"Downloaded: {dest}")
                return

    sys.exit(f"Attachment '{target}' listed in Gmail but data could not be fetched.")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "download"
    if mode not in ("download", "cleanup", "fetch", "text"):
        sys.exit("Usage: download_gmail_pdfs.py [download|cleanup|text|fetch <filename>]")

    if mode == "cleanup":
        cleanup(None)
        return

    creds = _get_credentials()
    service = build("gmail", "v1", credentials=creds)

    if mode == "fetch":
        if len(sys.argv) < 3:
            sys.exit("Usage: download_gmail_pdfs.py fetch <filename>")
        fetch_one(service, sys.argv[2])
    elif mode == "text":
        download_text(service)
    else:
        download(service)


if __name__ == "__main__":
    main()
