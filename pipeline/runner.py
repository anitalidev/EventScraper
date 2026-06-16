"""
Pipeline orchestrator.

Stages (in order):
  1. Scrape  — fetch raw posts from Instagram
  2. OCR     — extract text from post images (optional)
  3. Store   — persist raw posts before any AI processing
  4. Extract — AI extraction in configurable batches
  5. Validate — field-level validation + status assignment
  6. Store   — persist validated events with deduplication
"""
from __future__ import annotations
from datetime import date
from typing import Optional

import config
from extractors.base import BaseExtractor
from extractors.openai_extractor import OpenAIExtractor
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
    """
    Run the full pipeline and return a result summary dict.
    """
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

    # ── 3. Persist raw posts ─────────────────────────────────────────────────
    store.save_raw_posts(raw_posts)

    # ── 4. Extract in batches ────────────────────────────────────────────────
    extractor: BaseExtractor = OpenAIExtractor(api_key=api_key, model=model)
    all_extracted: list[ExtractedEvent] = []

    for batch in _chunk(raw_posts, batch_size):
        result["batches"] += 1
        try:
            extracted = extractor.extract_batch(batch)
            all_extracted.extend(extracted)
        except Exception as e:
            result["errors"].append({
                "batch": result["batches"],
                "error": str(e),
                "posts": [p.post_url for p in batch],
            })

    # ── 5. Validate ──────────────────────────────────────────────────────────
    validated = [validate(ev) for ev in all_extracted]
    for ev in validated:
        ev.source_label = "instagram"
    result["events_extracted"] = sum(1 for ev in validated if ev.is_event)

    # ── 6. Store events ──────────────────────────────────────────────────────
    counts = store.save_events(validated)
    result["storage_counts"] = counts

    # Serialise for the API response (exclude rejected non-events to reduce noise)
    result["events"] = [
        ev.to_dict()
        for ev in validated
        if ev.is_event
    ]

    return result
