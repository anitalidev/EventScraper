"""
Gmail scraper — fetches emails via the Gmail API and converts them to RawPost
objects so the existing extract → validate → store pipeline can process them
without any changes.
"""
from __future__ import annotations

import base64
import email.utils
import re
from datetime import datetime, timezone
from typing import Optional

from models.event import RawPost


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
        # Prefer text/plain parts; fall back to text/html stripped of tags
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


def _parse_date(date_header: str) -> str:
    """Parse an RFC 2822 date header into an ISO-8601 UTC string."""
    try:
        parsed = email.utils.parsedate_to_datetime(date_header)
        return parsed.astimezone(timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def fetch_raw_posts(service, max_results: int = 20, q: str = "") -> tuple[list[RawPost], list[str]]:
    """
    Fetch emails from Gmail and return (posts, errors).

    Each email becomes one RawPost:
      source   = "email"
      username = sender address
      post_url = Gmail deep-link to the message
      taken_at = sent date (UTC ISO-8601)
      caption  = "Subject: …\n\n<body text>"
    """
    posts: list[RawPost] = []
    errors: list[str] = []

    list_params: dict = {"userId": "me", "maxResults": max_results}
    if q:
        list_params["q"] = q

    try:
        results = service.users().messages().list(**list_params).execute()
    except Exception as e:
        errors.append(f"Gmail list failed: {e}")
        return posts, errors

    messages = results.get("messages", [])

    for msg in messages:
        try:
            data = service.users().messages().get(
                userId="me", id=msg["id"], format="full",
            ).execute()

            headers = {h["name"]: h["value"] for h in data["payload"].get("headers", [])}
            subject = headers.get("Subject", "(No Subject)")
            from_   = headers.get("From", "unknown")
            date_h  = headers.get("Date", "")

            body = _decode_body(data["payload"]).strip()
            if not body:
                body = data.get("snippet", "")

            caption = f"Subject: {subject}\n\n{body}"
            post_url = f"https://mail.google.com/mail/u/0/#inbox/{msg['id']}"

            posts.append(RawPost(
                source="email",
                username=from_,
                post_url=post_url,
                taken_at=_parse_date(date_h),
                caption=caption,
            ))
        except Exception as e:
            errors.append(f"Failed to fetch message {msg['id']}: {e}")

    return posts, errors
