"""
General event creation method.

All scrapers feed into create_events(). Sources are kept in memory only —
nothing is written to disk except final thumbnail crops.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field

import config
from extractors.thumbnail_extractor import generate_thumbnail
from ocr import processor as ocr
from storage import store


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Source:
    """One content item produced by any scraper. Never persisted to disk."""
    source: str                      # Instagram post URL, Gmail sender email, etc.
    text: str                        # Caption, email body, etc.
    image: Optional[str] = None      # Temp path to downloaded image
    ocr_text: Optional[str] = None   # Populated during the OCR pass if enabled


@dataclass
class Event:
    """One extracted event, fully built before being written to the DB."""
    source: str
    title: str
    date: Optional[str]
    time: Optional[str]
    description: str
    location: Optional[str]
    category_tags: list[str]
    thumbnail: Optional[str] = None  # Local path to cropped thumbnail, if any


# ── OpenAI response schema (internal) ────────────────────────────────────────

class _ExtractedEvent(BaseModel):
    source: str = Field(
        description="The exact source identifier of the content item this event was found in."
    )
    title: str = Field(description="Short, descriptive event title.")
    date: Optional[str] = Field(default=None, description="ISO date YYYY-MM-DD if found.")
    time: Optional[str] = Field(default=None, description="Start time HH:MM (24 h) if found.")
    description: str = Field(description="1-2 sentence summary of what the event is.")
    location: Optional[str] = Field(default=None, description="Venue or location string.")
    category_tags: list[str] = Field(description=(
        "One or more category tags. Choose only from: "
        "social, career, academic, arts, culture, outdoors, sports, food, wellness, volunteering."
    ))


class _BatchResponse(BaseModel):
    events: list[_ExtractedEvent]


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an event-extraction assistant for a university event aggregator.

You will receive a numbered list of content items. Each item has a source
identifier, text content, and optionally OCR text extracted from an image.

Extract ALL real, specific, upcoming events across all items. Return a flat
list under "events". One item may yield multiple events (e.g. a newsletter
listing several talks); extract each separately. Produce nothing for items
with no events.

Only extract events that:
- Have (or clearly imply) a specific date
- Are open to attendees

Do NOT extract:
- Job postings or hiring announcements
- Application or submission deadlines
- General club promotions with no specific date

Set "source" on each event to the exact source identifier of the item it came from.
For category_tags choose one or more from: social, career, academic, arts, culture,
outdoors, sports, food, wellness, volunteering."""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _dedupe_key(title: str, source: str, date: Optional[str], time: Optional[str]) -> str:
    combined = "|".join([
        title.lower().strip(),
        source.lower().strip(),
        (date or "") + (time or ""),
    ])
    return hashlib.sha1(combined.encode()).hexdigest()


def _chunk(lst: list, size: int) -> list[list]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def _build_user_message(batch: list[Source]) -> str:
    parts = []
    for i, src in enumerate(batch, 1):
        body = src.text
        if src.ocr_text:
            body += f"\n[OCR text: {src.ocr_text}]"
        parts.append(f"[{i}] Source: {src.source}\nText: {body}")
    return "\n\n---\n\n".join(parts)


# ── Pipeline stages ───────────────────────────────────────────────────────────

def _run_ocr(sources: list[Source]) -> None:
    """
    OCR pass — for every source that has an image, run Tesseract and store
    the result as ocr_text on that source. Mutates sources in-place.
    """
    for src in sources:
        if src.image:
            text = ocr.extract_text(src.image)
            if text:
                src.ocr_text = text


def _extract_batch(client: OpenAI, batch: list[Source], model: str) -> list[_ExtractedEvent]:
    """
    Send one batch of sources to OpenAI and return the raw extracted events.
    Returns an empty list if the API call fails.
    """
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_message(batch)},
            ],
            response_format=_BatchResponse,
            temperature=0,
        )
        parsed = response.choices[0].message.parsed
        return parsed.events if parsed else []
    except Exception as e:
        print(f"[event_creation] OpenAI batch failed: {e}")
        return []


def _build_events(
    extracted: list[_ExtractedEvent],
    source_to_image: dict[str, str],
    existing_keys: set[str],
    api_key: str,
    model: str,
) -> list[Event]:
    """
    Dedup, generate thumbnails, and build Event objects from one batch's
    extracted results. Updates existing_keys in-place to catch intra-run
    duplicates across batches.
    """
    events: list[Event] = []

    for item in extracted:
        key = _dedupe_key(item.title, item.source, item.date, item.time)
        if key in existing_keys:
            continue
        existing_keys.add(key)

        thumbnail: Optional[str] = None
        image_path = source_to_image.get(item.source)
        if image_path:
            thumbnail = generate_thumbnail(image_path, item.title, api_key=api_key, model=model)

        events.append(Event(
            source=item.source,
            title=item.title,
            date=item.date,
            time=item.time,
            description=item.description,
            location=item.location,
            category_tags=item.category_tags,
            thumbnail=thumbnail,
        ))

    return events


def _cleanup_batch_images(batch: list[Source]) -> None:
    """
    Delete all source images for this batch from local disk and clear
    their paths. Called after extraction and thumbnail generation are done.
    """
    for src in batch:
        if src.image:
            try:
                os.remove(src.image)
            except OSError:
                pass
            src.image = ""


def _save_events(events: list[Event], source_name: str) -> list[dict]:
    """
    Persist all events to the local SQLite DB and return the created rows.
    source_name ("Instagram", "Gmail", etc.) is added here, not earlier.
    """
    created: list[dict] = []

    for event in events:
        row = store.insert_scraped_event(
            dedupe_key=_dedupe_key(event.title, event.source, event.date, event.time),
            source_url=event.source,
            source_label=source_name,
            title=event.title,
            date=event.date,
            time=event.time,
            description=event.description,
            location=event.location,
            category_tags=event.category_tags,
            image_url=event.thumbnail,
        )
        if row:
            created.append(row)

    return created


# ── Entry point ───────────────────────────────────────────────────────────────

def create_events(
    sources: list[Source],
    source_name: str,
    api_key: str,
    *,
    ocr_enabled: bool = config.OCR_ENABLED,
    batch_size: int = config.BATCH_SIZE,
    model: str = config.OPENAI_MODEL,
) -> list[dict]:
    """
    General event creation method.

    Args:
        sources:     Content items from any scraper (in-memory only, never persisted).
        source_name: Human-readable label for the caller: "Instagram", "Gmail", etc.
        api_key:     OpenAI API key.
        ocr_enabled: Run Tesseract OCR on images before sending to OpenAI.
        batch_size:  Number of sources per OpenAI request.
        model:       OpenAI model to use.

    Returns:
        List of newly created event rows from the local DB.
    """
    if not sources:
        return []

    if ocr_enabled and ocr.is_available():
        _run_ocr(sources)

    client = OpenAI(api_key=api_key)
    existing_keys: set[str] = store.fetch_dedupe_keys()
    current_events: list[Event] = []

    for batch in _chunk(sources, batch_size):
        source_to_image: dict[str, str] = {
            src.source: src.image for src in batch if src.image
        }
        extracted = _extract_batch(client, batch, model)
        batch_events = _build_events(extracted, source_to_image, existing_keys, api_key, model)
        current_events.extend(batch_events)
        _cleanup_batch_images(batch)

    return _save_events(current_events, source_name)
