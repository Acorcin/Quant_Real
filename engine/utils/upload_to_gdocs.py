"""
Google Docs Uploader Utility.

Reads markdown audit reports from the `docs/` directory, converts them to HTML,
and uploads them to Google Drive, converting them into Google Documents.

Prerequisites:
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib markdown

Setup:
    1. Go to Google Cloud Console (https://console.cloud.google.com).
    2. Create a project, enable the Google Drive API.
    3. Configure OAuth Consent Screen (Internal or External, add your email).
    4. Create Credentials -> OAuth client ID -> Application type: Desktop app.
    5. Download JSON, rename it to `credentials.json`, and place it in the project root.
    6. Run this script: `python utils/upload_to_gdocs.py`
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("gdocs_uploader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    import markdown
except ImportError:
    logger.error(
        "Missing required libraries. Please run:\n"
        "  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib markdown"
    )
    sys.exit(1)

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

def authenticate_google_drive():
    """Authenticates the user and returns the Drive API service client."""
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.warning(f"Failed to refresh token: {e}. Re-authenticating...")
                creds = None
        
        if not creds:
            if not os.path.exists("credentials.json"):
                logger.error(
                    "credentials.json not found!\n"
                    "Please download OAuth client ID credentials from Google Cloud Console "
                    "and save them as credentials.json in the project root directory."
                )
                return None
            
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
            
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())
            
    return build("drive", "v3", credentials=creds)

def upload_markdown_as_doc(service, file_path: Path) -> str | None:
    """Reads a markdown file, converts it to HTML, and uploads to Google Drive as a Google Doc."""
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None

    logger.info(f"Converting and uploading {file_path.name} to Google Docs...")

    # Read markdown file
    with open(file_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Convert markdown to basic HTML (Google Docs Drive converter imports HTML very well)
    # Enable table extensions to preserve markdown tables
    html_content = markdown.markdown(
        md_content,
        extensions=["tables", "fenced_code", "codehilite"]
    )

    # Wrap in basic HTML structure with UTF-8 meta header for correct character encoding
    full_html = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head><meta charset='utf-8'></head>\n"
        f"<body>\n{html_content}\n</body>\n"
        "</html>"
    )

    # Convert HTML string to generic byte stream
    import io
    fh = io.BytesIO(full_html.encode("utf-8"))
    
    # Create file metadata
    # Setting mimeType to 'application/vnd.google-apps.document' instructs Drive
    # to automatically convert our HTML upload into a native Google Document.
    file_metadata = {
        "name": file_path.stem.replace("audit_", "Audit: ").replace("_", " ").title(),
        "mimeType": "application/vnd.google-apps.document"
    }

    media = MediaIoBaseUpload(
        fh,
        mimetype="text/html",
        resumable=True
    )

    try:
        # Check if file with same name already exists in Drive to update it,
        # or upload it as a new file.
        query = f"name = '{file_metadata['name']}' and mimeType = 'application/vnd.google-apps.document' and trashed = false"
        results = service.files().list(q=query, spaces="drive", fields="files(id)").execute()
        items = results.get("files", [])
        
        if items:
            # Update existing file
            file_id = items[0]["id"]
            updated_file = service.files().update(
                fileId=file_id,
                media_body=media
            ).execute()
            logger.info(f"Successfully updated Google Doc: '{file_metadata['name']}' (ID: {file_id})")
            return file_id
        else:
            # Upload new file
            new_file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id"
            ).execute()
            file_id = new_file.get("id")
            logger.info(f"Successfully created Google Doc: '{file_metadata['name']}' (ID: {file_id})")
            return file_id
            
    except Exception as e:
        logger.error(f"Error uploading to Google Drive: {e}")
        return None

def main():
    service = authenticate_google_drive()
    if not service:
        logger.error("Authentication failed. Aborting.")
        return

    # Find all generated audit reports in docs/
    project_root = Path(__file__).resolve().parent.parent
    docs_dir = project_root / "docs"
    
    files_to_upload = [
        docs_dir / "audit_agent1_quant_strategy.md",
        docs_dir / "audit_agent2_architecture.md",
        docs_dir / "audit_agent3_execution.md",
        docs_dir / "audit_agent4_signals.md",
        docs_dir / "audit_agent5_closed_loop.md",
        docs_dir / "consolidated_audit_report.md"
    ]

    uploaded_ids = {}
    for f_path in files_to_upload:
        if f_path.exists():
            doc_id = upload_markdown_as_doc(service, f_path)
            if doc_id:
                uploaded_ids[f_path.name] = doc_id

    if uploaded_ids:
        logger.info("=" * 60)
        logger.info("UPLOAD COMPLETE. Access your files on Google Docs:")
        for name, doc_id in uploaded_ids.items():
            url = f"https://docs.google.com/document/d/{doc_id}/edit"
            logger.info(f"  - {name}: {url}")
        logger.info("=" * 60)

if __name__ == "__main__":
    main()
