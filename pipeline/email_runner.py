"""
Email pipeline — fetch → extract → validate → store.

Mirrors pipeline/runner.py but skips the Instagram scraping and OCR steps.
"""
from __future__ import annotations

import config
from extractors.base import ExtractionUnavailableError
from extractors.email_extractor import EmailExtractor
from models.event import RawPost
from pipeline.batching import extract_with_bisection, iter_bounded_batches
from storage import store
from validation.validator import validate


def _input_size(post: RawPost) -> int:
    # Include a small allowance for JSON keys and metadata.
    return len(post.caption) + len(post.taken_at) + 96


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

    store.save_raw_posts(raw_posts)

    extractor = EmailExtractor(api_key=api_key, model=model)

    all_extracted = []
    for batch in iter_bounded_batches(
        raw_posts,
        max_items=batch_size,
        max_chars=config.BATCH_MAX_INPUT_CHARS,
        size_of=_input_size,
    ):
        extracted, failures, calls = extract_with_bisection(
            batch,
            extractor.extract_batch,
            should_split=lambda error: not isinstance(
                error, ExtractionUnavailableError
            ),
        )
        result["batches"] += calls
        all_extracted.extend(extracted)
        for failure in failures:
            result["errors"].append({
                "error": str(failure.error),
                "posts": [p.post_url for p in failure.items],
            })

    validated = [validate(ev) for ev in all_extracted]
    for ev in validated:
        ev.source_label = "Newsletter"
    result["events_extracted"] = sum(1 for ev in validated if ev.is_event)

    counts = store.save_events(validated)
    result["storage_counts"] = counts

    result["events"] = [ev.to_dict() for ev in validated if ev.is_event]

    return result
