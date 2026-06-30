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
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from PIL import Image

import config
from storage.store import find_raw_post_image

log = logging.getLogger(__name__)


def _upload_image(event_id: str, local_path: str) -> bool:
    """Upload an event image through the UBC Discovery presigned POST flow."""
    headers = {"Authorization": f"Api-Key {config.UBC_DISCOVERY_API_KEY}"}
    presign_url = (
        config.UBC_DISCOVERY_API_URL.rstrip("/")
        + f"/events/{event_id}/presigned-upload"
    )

    try:
        presign_resp = requests.post(
            presign_url,
            headers=headers,
            timeout=15,
        )
        if not presign_resp.ok:
            raise UBCDiscoveryError(
                status_code=presign_resp.status_code,
                body=presign_resp.text,
            )

        upload = presign_resp.json()
        content_type = upload["fields"].get(
            "Content-Type",
            mimetypes.guess_type(local_path)[0] or "image/jpeg",
        )
        image, filename = _prepare_image(local_path, content_type)
        max_size = upload["max_file_size_bytes"]
        file_size = image.getbuffer().nbytes
        if file_size > max_size:
            log.warning(
                "Event image %s is too large for UBC Discovery (%d > %d bytes)",
                local_path,
                file_size,
                max_size,
            )
            return False

        upload_resp = requests.post(
            upload["upload_url"],
            data=upload["fields"],
            files={"file": (filename, image, content_type)},
            timeout=30,
        )
        upload_resp.raise_for_status()
        log.info(
            "Uploaded event image for UBC Discovery event %s: %s",
            event_id,
            upload["file_key"],
        )
        return True
    except Exception as e:
        log.warning(
            "Failed to upload image %s for UBC Discovery event %s: %s",
            local_path,
            event_id,
            e,
        )
        return False


def _prepare_image(local_path: str, content_type: str) -> tuple[BytesIO, str]:
    """Return image bytes matching the content type required by the presigned POST."""
    output = BytesIO()
    if content_type == "image/webp":
        with Image.open(local_path) as source:
            source.save(output, format="WEBP")
        filename = os.path.splitext(os.path.basename(local_path))[0] + ".webp"
    else:
        with open(local_path, "rb") as source:
            output.write(source.read())
        filename = os.path.basename(local_path)
    output.seek(0)
    return output, filename


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
    """Interpret an event's local date/time and return a UTC ISO-8601 string."""
    if not date_str:
        return None
    dt_str = f"{date_str}T{time_str}:00" if time_str else f"{date_str}T00:00:00"
    local_timezone = timezone(timedelta(hours=config.APP_UTC_OFFSET_HOURS))
    local_datetime = datetime.fromisoformat(dt_str).replace(tzinfo=local_timezone)
    return local_datetime.astimezone(timezone.utc).isoformat()


def _build_payload(event: dict) -> dict:
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
    }


def _strip_nones(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def _local_image_path(event: dict) -> Optional[str]:
    image_url = event.get("image_url") or ""
    if image_url.startswith("/api/images/"):
        filename = os.path.basename(image_url.removeprefix("/api/images/"))
        local_image = os.path.join(config.IMG_DIR, filename)
        if os.path.isfile(local_image):
            return local_image

    raw_image = find_raw_post_image(
        post_url=event.get("source_url") or "",
        username=event.get("organizer") or "",
    )
    return raw_image if raw_image and os.path.isfile(raw_image) else None


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

    local_image = _local_image_path(event)
    payload = _strip_nones(_build_payload(event))

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
    if local_image:
        _upload_image(data["id"], local_image)

    return CreatedEvent(
        ubc_event_id=data["id"],
        title=data["title"],
        created_at=data["created_at"],
    )
