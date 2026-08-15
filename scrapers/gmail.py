"""
Gmail scraper.

Uses a pre-authenticated Gmail API service object (Google OAuth) to fetch
emails addressed to the +event alias, builds a list of Source objects,
and delegates to the generalized pipeline.
"""
from __future__ import annotations

import base64
import email.utils
import os
import re
from datetime import datetime, timezone
from typing import Optional

import config
from pipeline.event_creation import Source, create_events

_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def _decode_body(payload: dict) -> str:
    """Recursively extract plain-text body from a Gmail message payload."""
    mime = payload.get("mimeType", "")

    if mime == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
        return ""

    if mime.startswith("multipart/"):
        parts = payload.get("parts", [])
        plain = ""
        html_fallback = ""
        for part in parts:
            part_mime = part.get("mimeType", "")
            if part_mime == "text/plain":
                plain += _decode_body(part)
            elif part_mime == "text/html" and not plain:
                raw = _decode_body(part)
                html_fallback += re.sub(r"<[^>]+>", " ", raw)
            elif part_mime.startswith("multipart/"):
                plain += _decode_body(part)
        return plain or html_fallback

    return ""


def _first_image_attachment(service, message_id: str, payload: dict) -> Optional[str]:
    """
    Download the first image attachment from the message and save it locally.
    Returns the file path, or None if there are no image attachments.
    """
    for part in payload.get("parts", []):
        mime = part.get("mimeType", "")
        if mime not in _IMAGE_MIME_TYPES:
            continue

        attachment_id = part.get("body", {}).get("attachmentId")
        if not attachment_id:
            continue

        try:
            attachment = service.users().messages().attachments().get(
                userId="me", messageId=message_id, id=attachment_id
            ).execute()
            data = base64.urlsafe_b64decode(attachment["data"] + "==")

            ext = mime.split("/")[-1].replace("jpeg", "jpg")
            os.makedirs(config.IMG_DIR, exist_ok=True)
            path = os.path.join(config.IMG_DIR, f"gmail_{message_id}.{ext}")
            with open(path, "wb") as f:
                f.write(data)
            return path
        except Exception as e:
            print(f"[gmail] attachment download failed for {message_id}: {e}")

    return None


def _parse_sender_email(from_header: str) -> str:
    """Extract the bare email address from a From header like 'Name <addr@example.com>'."""
    _, addr = email.utils.parseaddr(from_header)
    return addr or from_header


def scrape_gmail(
    service,
    api_key: str,
    *,
    max_results: int = 20,
    ocr_enabled: bool = config.OCR_ENABLED,
    batch_size: int = config.BATCH_SIZE,
    model: str = config.OPENAI_MODEL,
) -> tuple[list[dict], int]:
    """
    Fetch emails from Gmail (filtered to the +event alias) and extract events.

    Args:
        service:     Authenticated Gmail API service object.
        api_key:     OpenAI API key.
        max_results: Maximum number of emails to fetch.

    Returns:
        (events, emails_scraped) — list of created event rows and the count
        of emails that were successfully fetched.
    """
    sources: list[Source] = []

    try:
        results = service.users().messages().list(
            userId="me",
            maxResults=max_results,
            q="to:+event",
        ).execute()
    except Exception as e:
        print(f"[gmail] list failed: {e}")
        return [], 0

    for msg in results.get("messages", []):
        try:
            data = service.users().messages().get(
                userId="me", id=msg["id"], format="full",
            ).execute()

            headers = {h["name"]: h["value"] for h in data["payload"].get("headers", [])}
            subject = headers.get("Subject", "(No Subject)")
            from_header = headers.get("From", "")
            sender_email = _parse_sender_email(from_header)

            body = _decode_body(data["payload"]).strip() or data.get("snippet", "")
            text = f"Subject: {subject}\n\n{body}"

            image_path = _first_image_attachment(service, msg["id"], data["payload"])

            sources.append(Source(
                source=sender_email,
                text=text,
                image=image_path,
            ))
        except Exception as e:
            print(f"[gmail] failed to fetch message {msg['id']}: {e}")

    emails_scraped = len(sources)
    events = create_events(sources, "Gmail", api_key,
                    ocr_enabled=ocr_enabled, batch_size=batch_size, model=model)
    return events, emails_scraped
