"""
Client for the UBC Discovery events API.

Called only after a human approves an event in the review dashboard.
Never called automatically — human approval is the gate.
"""
from __future__ import annotations

import json
import logging
import mimetypes
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import boto3
import requests
from botocore.config import Config

import config

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _s3_client():
    return boto3.client(
        "s3",
        region_name=config.AWS_REGION,
        endpoint_url=config.S3_ENDPOINT_URL or None,
        config=Config(signature_version="s3v4"),
    )


def _upload_image(local_path: str) -> Optional[str]:
    """Upload a local image file to S3 and return the object key, or None on failure."""
    if not config.S3_BUCKET_NAME:
        log.warning("S3_BUCKET_NAME not configured — skipping image upload")
        return None

    ext = os.path.splitext(local_path)[1] or ".jpg"
    key = f"event-images/{uuid.uuid4()}{ext}"
    content_type = mimetypes.guess_type(local_path)[0] or "image/jpeg"

    try:
        with open(local_path, "rb") as f:
            _s3_client().put_object(
                Bucket=config.S3_BUCKET_NAME,
                Key=key,
                Body=f,
                ContentType=content_type,
            )
        log.info("Uploaded event image to S3: %s", key)
        os.remove(local_path)
        log.info("Deleted local image: %s", local_path)
        return key
    except Exception as e:
        log.warning("Failed to upload image %s to S3: %s", local_path, e)
        return None


class UBCDiscoveryError(Exception):
    """Raised when UBC Discovery returns a non-2xx response."""
    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"UBC Discovery API error {status_code}: {body}")


class UBCDiscoveryConflict(UBCDiscoveryError):
    """Raised on 409 — event already exists (keyed on external_ref)."""
    def __init__(self, existing_id: str, body: str):
        self.existing_id = existing_id
        super().__init__(409, body)


@dataclass
class CreatedEvent:
    ubc_event_id: str   # nanoid string, e.g. "aB3xZ9qR"
    title: str
    created_at: str


def _combine_datetime(date_str: Optional[str], time_str: Optional[str]) -> Optional[str]:
    """Combine YYYY-MM-DD + HH:MM into a UTC ISO-8601 datetime string."""
    if not date_str:
        return None
    dt_str = f"{date_str}T{time_str}:00" if time_str else f"{date_str}T00:00:00"
    return datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc).isoformat()


def _build_payload(event: dict, event_picture_key: Optional[str] = None) -> dict:
    """Map an EventScraper event dict to the UBC Discovery CreateEventRequest body."""
    return {
        "title":             event["title"],
        "description":       event.get("description") or "",
        "club_name":         event.get("organizer"),
        "source":            "instagram",
        "source_label":      "ams_club",
        "source_url":        event.get("source_url"),
        "vibes":             json.loads(event.get("vibes") or "[]"),
        "location_name":     event.get("location"),
        "event_date":        _combine_datetime(event.get("date"), event.get("time")),
        "external_ref":      str(event["id"]),  # idempotency key (requires UBC Discovery change)
        "event_picture_key": event_picture_key,
    }


def _strip_nones(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def list_events(page_size: int = 20) -> list[dict]:
    """
    Fetch all events from UBC Discovery using GET /events?skip=0&limit=20.
    Paginates until exhausted. Public endpoint — no auth required.
    Returns an empty list if UBC_DISCOVERY_API_URL is not configured.
    """
    if not config.UBC_DISCOVERY_API_URL:
        return []

    base_url = config.UBC_DISCOVERY_API_URL.rstrip("/") + "/events"
    all_events: list[dict] = []
    skip = 0

    while True:
        try:
            resp = requests.get(
                base_url,
                params={"skip": skip, "limit": page_size},
                headers={"Accept": "application/json"},
                timeout=15,
            )
            resp.raise_for_status()
            page = resp.json()
        except Exception as e:
            log.warning("Could not fetch UBC Discovery events (skip=%d): %s", skip, e)
            break

        items = page if isinstance(page, list) else page.get("events", [])

        if not items:
            break

        all_events.extend(items)

        if len(items) < page_size:
            break

        skip += page_size

    return all_events


def publish_event(event: dict) -> CreatedEvent:
    """
    POST the approved event to UBC Discovery.

    Raises:
        ValueError           — API URL or key not configured.
        UBCDiscoveryConflict — event already exists (caller may treat as success).
        UBCDiscoveryError    — any other API failure.
        requests.Timeout     — network timeout.
    """
    if not config.UBC_DISCOVERY_API_URL:
        raise ValueError("UBC_DISCOVERY_API_URL is not configured")
    if not config.UBC_DISCOVERY_API_KEY:
        raise ValueError("UBC_DISCOVERY_API_KEY is not configured")

    url = config.UBC_DISCOVERY_API_URL.rstrip("/") + "/events"

    event_picture_key: Optional[str] = None
    image_url = event.get("image_url") or ""
    if image_url.startswith("/api/images/"):
        filename = image_url.removeprefix("/api/images/")
        local_image = os.path.join(config.IMG_DIR, filename)
        if os.path.isfile(local_image):
            event_picture_key = _upload_image(local_image)

    payload = _strip_nones(_build_payload(event, event_picture_key))

    log.info("Publishing event %s to UBC Discovery: %s", event["id"], event["title"])

    resp = requests.post(
        url,
        json=payload,
        headers={
            "Authorization": f"Api-Key {config.UBC_DISCOVERY_API_KEY}",
            "Content-Type": "application/json",
        },
        timeout=15,
    )

    if resp.status_code == 409:
        body = resp.json() if resp.content else {}
        existing_id = body.get("existing_id", "")
        raise UBCDiscoveryConflict(existing_id=existing_id, body=resp.text)

    if not resp.ok:
        raise UBCDiscoveryError(status_code=resp.status_code, body=resp.text)

    data = resp.json()
    log.info("UBC Discovery created event id=%s for EventScraper id=%s", data["id"], event["id"])
    return CreatedEvent(
        ubc_event_id=data["id"],
        title=data["title"],
        created_at=data["created_at"],
    )
