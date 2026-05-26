"""Download PDF attachments from Gmail into data/pdfs/."""
import base64
import sys
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

CLIENT_SECRET = Path.home() / ".config/gws/client_secret.json"
TOKEN_PATH = Path.home() / ".config/gws/gmail_token.json"
DEST_DIR = Path("data/pdfs")
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


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
    response = (
        service.users().messages().list(userId="me", q="has:attachment filename:pdf").execute()
    )
    while True:
        msg_ids.extend(m["id"] for m in response.get("messages", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
        response = service.users().messages().list(
            userId="me", q="has:attachment filename:pdf", pageToken=page_token
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


def main() -> None:
    creds = _get_credentials()
    service = build("gmail", "v1", credentials=creds)

    msg_ids = _list_all_messages(service)
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    found = skipped = downloaded = 0

    for msg_id in msg_ids:
        message = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
        parts = _pdf_parts(message["payload"])
        for part in parts:
            filename = part.get("filename", "")
            if not filename.lower().endswith(".pdf"):
                continue
            found += 1
            dest = DEST_DIR / filename
            if dest.exists():
                skipped += 1
                continue
            data = _download_attachment(service, msg_id, part)
            dest.write_bytes(data)
            downloaded += 1

    print(
        f"{found} PDF attachments found, {skipped} skipped (already present), "
        f"{downloaded} downloaded into data/pdfs/"
    )


if __name__ == "__main__":
    main()
