"""
Pipeline orchestrator.

Stages (in order):
  1. Scrape   — fetch content from Instagram
  2. OCR      — extract text from images (optional)
  3. Extract  — AI extraction in configurable batches → flat list of events
  4. Validate — field-level validation + status assignment
  5. Thumbnails — generate thumbnails for accepted events
  6. Store    — persist validated events with deduplication
"""
from __future__ import annotations
import os
from datetime import date

import config
from extractors.openai_extractor import OpenAIExtractor
from extractors.thumbnail_extractor import generate_thumbnail
from models.event import ExtractedEvent, RawPost
from ocr import processor as ocr
from scrapers.instagram import scrape_channels
from storage import store
from validation.validator import validate


def _chunk(lst: list, size: int) -> list[list]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def run(
    channels: list[str],
    start_date: date,
    end_date: date,
    api_key: str,
    *,
    batch_size: int = config.BATCH_SIZE,
    ocr_enabled: bool = config.OCR_ENABLED,
    model: str = config.OPENAI_MODEL,
) -> dict:
    result: dict = {
        "posts_scraped": 0,
        "posts_with_ocr": 0,
        "batches": 0,
        "events_extracted": 0,
        "storage_counts": {},
        "events": [],
        "errors": [],
    }

    # ── 1. Scrape ────────────────────────────────────────────────────────────
    raw_posts, scrape_errors = scrape_channels(channels, start_date, end_date)
    result["posts_scraped"] = len(raw_posts)
    result["errors"].extend(scrape_errors)

    if not raw_posts:
        return result

    # ── 2. OCR ───────────────────────────────────────────────────────────────
    if ocr_enabled and ocr.is_available():
        for post in raw_posts:
            text = ocr.extract_text(post.image_path)
            if text:
                post.ocr_text = text
                result["posts_with_ocr"] += 1

    # Build URL → image_path lookup for thumbnail generation later
    url_to_image: dict[str, str] = {
        p.post_url: p.image_path
        for p in raw_posts
        if p.image_path
    }

    # ── 3. Extract in batches ────────────────────────────────────────────────
    extractor = OpenAIExtractor(api_key=api_key, model=model)
    all_extracted: list[ExtractedEvent] = []

    for batch in _chunk(raw_posts, batch_size):
        result["batches"] += 1
        try:
            all_extracted.extend(extractor.extract_batch(batch))
        except Exception as e:
            result["errors"].append({
                "batch": result["batches"],
                "error": str(e),
                "urls": [p.post_url for p in batch],
            })

    # ── 4. Validate ──────────────────────────────────────────────────────────
    validated = [validate(ev) for ev in all_extracted]
    for ev in validated:
        ev.source_label = "instagram"
    result["events_extracted"] = sum(1 for ev in validated if ev.is_event)

    # ── 5. Thumbnails ────────────────────────────────────────────────────────
    for ev in validated:
        image_path = url_to_image.get(ev.source_url)
        if ev.is_event and image_path:
            ev.image_url = generate_thumbnail(
                image_path,
                ev.title,
                api_key=api_key,
                model=model,
            )

    # ── 6. Discard source images ─────────────────────────────────────────────
    for image_path in url_to_image.values():
        try:
            os.remove(image_path)
        except OSError:
            pass

    # ── 7. Store events ──────────────────────────────────────────────────────
    counts = store.save_events(validated)
    result["storage_counts"] = counts

    result["events"] = [
        ev.to_dict()
        for ev in validated
        if ev.is_event and not ev.is_duplicate
    ]

    return result
