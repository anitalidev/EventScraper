"""
Validation layer — applied to every ExtractedEvent before it is written to
storage or returned to the UI.

Rules are deliberately strict: bad data should be rejected or flagged rather
than silently passed through.
"""
from __future__ import annotations
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

import config
from models.event import ExtractedEvent, VIBE_VALUES

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_RE = re.compile(r"^\d{2}:\d{2}$")
_URL_RE  = re.compile(r"^https?://")


def validate(
    ev: ExtractedEvent,
    *,
    current_date: date | None = None,
) -> ExtractedEvent:
    """
    Validate *ev* in-place.  Appends human-readable strings to
    ev.validation_errors and updates ev.status accordingly.
    Returns the same object.
    """
    errors: list[str] = []

    # ── field-level checks ──────────────────────────────────────────────────
    if not ev.title or not ev.title.strip():
        errors.append("title is empty")

    if not ev.source_url or not _URL_RE.match(ev.source_url):
        errors.append(f"source_url is missing or not a URL: {ev.source_url!r}")

    if ev.date is not None:
        if not _DATE_RE.match(ev.date):
            errors.append(f"date is not ISO YYYY-MM-DD: {ev.date!r}")
        else:
            try:
                event_date = date.fromisoformat(ev.date)
            except ValueError:
                errors.append(f"date is not a real calendar date: {ev.date!r}")
            else:
                today = current_date or datetime.now(
                    ZoneInfo(config.APP_TIMEZONE)
                ).date()
                if event_date < today:
                    errors.append(f"event date is in the past: {ev.date}")

    if ev.time is not None and not _TIME_RE.match(ev.time):
        errors.append(f"time is not HH:MM: {ev.time!r}")

    if not (0.0 <= ev.confidence <= 1.0):
        errors.append(f"confidence {ev.confidence} is outside [0, 1]")

    ev.vibes = [v for v in ev.vibes if v in VIBE_VALUES] or ["social"]

    ev.validation_errors = errors

    # ── status assignment ────────────────────────────────────────────────────
    if errors or not ev.is_event or ev.confidence < config.CONFIDENCE_REVIEW:
        ev.status = "rejected"
    else:
        ev.status = "review"

    return ev
