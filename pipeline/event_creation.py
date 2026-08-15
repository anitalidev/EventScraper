"""
General event creation method.

All scrapers feed into create_events(). Sources are kept in memory only —
nothing is written to disk except final thumbnail crops.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
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
    source: str                      # Instagram post URL, Gmail message link, etc.
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
    thumbnail: Optional[str] = None  # Local path to cropped thumbnail image, if any


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


# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── Main function ─────────────────────────────────────────────────────────────

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

    # ── 1. OCR pass ───────────────────────────────────────────────────────────
    if ocr_enabled and ocr.is_available():
        for src in sources:
            if src.image:
                text = ocr.extract_text(src.image)
                if text:
                    src.ocr_text = text

    # ── 2. Extract events in batches ──────────────────────────────────────────
    client = OpenAI(api_key=api_key)

    # Fetch all existing dedupe keys once. New keys are added as events are
    # accepted, so intra-run duplicates across batches are also caught.
    existing_keys: set[str] = store.fetch_dedupe_keys()
    current_events: list[Event] = []

    for batch in _chunk(sources, batch_size):
        # Call OpenAI for this batch
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
        except Exception as e:
            print(f"[scrape] OpenAI batch failed: {e}")
            parsed = None

        if parsed is not None:
            # Map source identifier → image path for this batch
            source_to_image: dict[str, str] = {
                src.source: src.image
                for src in batch
                if src.image
            }

            for extracted in parsed.events:
                key = _dedupe_key(extracted.title, extracted.source, extracted.date, extracted.time)
                if key in existing_keys:
                    continue
                existing_keys.add(key)

                # Generate thumbnail now, while the image is still on disk
                thumbnail: Optional[str] = None
                image_path = source_to_image.get(extracted.source)
                if image_path:
                    thumbnail = generate_thumbnail(
                        image_path, extracted.title, api_key=api_key, model=model
                    )

                current_events.append(Event(
                    source=extracted.source,
                    title=extracted.title,
                    date=extracted.date,
                    time=extracted.time,
                    description=extracted.description,
                    location=extracted.location,
                    category_tags=extracted.category_tags,
                    thumbnail=thumbnail,
                ))

        # Delete every source image in this batch now that we're done with it
        for src in batch:
            if src.image:
                try:
                    os.remove(src.image)
                except OSError:
                    pass
                src.image = ""

    # ── 3. Save events to local DB ────────────────────────────────────────────
    created: list[dict] = []

    for event in current_events:
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
