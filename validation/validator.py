"""
Validation layer — applied to every ExtractedEvent before it is written to
storage or returned to the UI.

Rules are deliberately strict: bad data should be rejected or flagged rather
than silently passed through.
"""
from __future__ import annotations
import re
from datetime import date

import config
from models.event import ExtractedEvent

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_RE = re.compile(r"^\d{2}:\d{2}$")
_URL_RE  = re.compile(r"^https?://")


def validate(ev: ExtractedEvent) -> ExtractedEvent:
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
                date.fromisoformat(ev.date)
            except ValueError:
                errors.append(f"date is not a real calendar date: {ev.date!r}")

    if ev.time is not None and not _TIME_RE.match(ev.time):
        errors.append(f"time is not HH:MM: {ev.time!r}")

    if not (0.0 <= ev.confidence <= 1.0):
        errors.append(f"confidence {ev.confidence} is outside [0, 1]")

    if ev.event_type not in config.EVENT_TYPES:
        # Normalise unknown types to "Other" rather than hard-rejecting
        ev.event_type = "Other"

    ev.validation_errors = errors

    # ── status assignment ────────────────────────────────────────────────────
    if errors:
        ev.status = "rejected"
    elif not ev.is_event:
        ev.status = "rejected"
    elif ev.confidence >= config.CONFIDENCE_PUBLISH:
        ev.status = "published"
    elif ev.confidence >= config.CONFIDENCE_REVIEW:
        ev.status = "review"
    else:
        ev.status = "rejected"

    return ev
