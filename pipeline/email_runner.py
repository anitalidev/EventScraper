"""
Email pipeline — fetch → extract → validate → store.

Mirrors pipeline/runner.py but skips the Instagram scraping and OCR steps.
"""
from __future__ import annotations

import config
from extractors.email_extractor import EmailExtractor
from models.event import RawPost
from storage import store
from validation.validator import validate



def _chunk(lst: list, size: int) -> list[list]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def run_email(
    raw_posts: list[RawPost],
    api_key: str,
    *,
    batch_size: int = config.BATCH_SIZE,
    model: str = config.OPENAI_MODEL,
) -> dict:
    """
    Run the email pipeline on already-fetched RawPost objects.
    Returns the same result shape as pipeline.runner.run().
    """
    result: dict = {
        "posts_scraped": len(raw_posts),
        "posts_with_ocr": 0,
        "batches": 0,
        "events_extracted": 0,
        "storage_counts": {},
        "events": [],
        "errors": [],
    }

    if not raw_posts:
        return result

    extractor = EmailExtractor(api_key=api_key, model=model)

    all_extracted = []
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

    validated = [validate(ev) for ev in all_extracted]
    for ev in validated:
        ev.source_label = "Newsletter"
    result["events_extracted"] = sum(1 for ev in validated if ev.is_event)

    counts = store.save_events(validated)
    result["storage_counts"] = counts

    result["events"] = [ev.to_dict() for ev in validated if ev.is_event and not ev.is_duplicate]

    return result
