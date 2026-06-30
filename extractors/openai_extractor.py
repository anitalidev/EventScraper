"""
OpenAI-backed extractor using structured JSON outputs (response_format).

Structured outputs guarantee the model returns valid JSON that matches our
Pydantic schema, eliminating the need for manual parsing or error-prone regex
post-processing.

Retry strategy: exponential back-off on rate-limit / server errors (tenacity).
"""
from __future__ import annotations
import json
import time
from typing import Optional

from openai import OpenAI, RateLimitError, APIStatusError
from pydantic import BaseModel, Field

import config
from extractors.base import BaseExtractor, ExtractionUnavailableError
from models.event import ExtractedEvent, RawPost, Vibe

# ── Pydantic schema that OpenAI will enforce ────────────────────────────────

class _SingleEvent(BaseModel):
    input_id: int = Field(description="The input_id copied exactly from the input post.")
    is_event: bool = Field(description=(
        "Whether the post qualifies as an event under the system rules."
    ))
    confidence: float = Field(ge=0.0, le=1.0, description=(
        "Confidence in the is_event classification."
    ))
    title: str = Field(description="Event name, extract from post")
    date: Optional[str] = Field(default=None, description="ISO date YYYY-MM-DD if found.")
    time: Optional[str] = Field(default=None, description="Start time HH:MM (24 h) if found.")
    location: Optional[str] = Field(default=None, description="Event venue or location")
    description: str = Field(description="1-2 sentence summary of what the event is.")
    vibes: list[Vibe] = Field(
        min_length=1,
        max_length=3,
        description="Best matching event categories.",
    )


class _BatchResponse(BaseModel):
    results: list[_SingleEvent]


# ── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """Analyse Instagram post captions and decide whether each post
describes a real, specific event.

REJECT (set is_event=false) for:
- Job postings or hiring announcements
- Applications open / deadline reminders
- Executive or director recruitment
- General club promotions without a specific event date
- Announcements that do not correspond to a single, specific event occurrence

ACCEPT (set is_event=true) only when the post describes an event that:
- Has (or implies) a specific date
- Is open to attendees
- Is not just a recurring programme announcement with no specifics

Resolve relative dates (such as "tomorrow" or "this Friday") and dates without
a year using that post's posted_date. Extract qualifying events whether their
date is past, present, or future; downstream validation handles recency.

Assign confidence based on how many concrete details are present:
- 0.9+ : date, time, location, and clear title all present
- 0.7-0.9 : date present plus at least one of time / location
- 0.5-0.7 : only a date is inferable
- below 0.5 : very uncertain — likely not a real event

Return exactly one result for each input post. Copy its input_id exactly into
the result. Do not omit, duplicate, or invent input IDs.
"""


class OpenAIExtractor(BaseExtractor):
    """
    Extracts events from batches of RawPost objects using the OpenAI API
    with structured JSON output mode.
    """

    def __init__(self, api_key: str, model: str = config.OPENAI_MODEL):
        # Retry here so retry accounting and request-wide failure handling stay
        # visible to the pipeline rather than stacking with SDK retries.
        self._client = OpenAI(api_key=api_key, max_retries=0)
        self._model = model

    def extract_batch(self, posts: list[RawPost]) -> list[ExtractedEvent]:
        if not posts:
            return []

        user_msg = self._build_user_message(posts)
        raw_response = self._call_api_with_retry(user_msg)

        posts_by_id = {
            input_id: post for input_id, post in enumerate(posts)
        }
        response_ids = [item.input_id for item in raw_response.results]
        if len(response_ids) != len(set(response_ids)):
            raise ValueError("Model response contains duplicate input IDs.")
        if set(response_ids) != set(posts_by_id):
            missing = sorted(set(posts_by_id) - set(response_ids))
            unknown = sorted(set(response_ids) - set(posts_by_id))
            raise ValueError(
                f"Model response input IDs do not match request "
                f"(missing={missing}, unknown={unknown})."
            )

        response_by_id = {
            item.input_id: item for item in raw_response.results
        }
        results: list[ExtractedEvent] = []
        for input_id, post in posts_by_id.items():
            item = response_by_id[input_id]
            ev = ExtractedEvent(
                is_event=item.is_event,
                confidence=item.confidence,
                title=item.title,
                date=item.date,
                time=item.time,
                location=item.location,
                description=item.description,
                source_url=post.post_url,
                organizer=post.username,
                vibes=item.vibes,
                raw_ai_response=item.model_dump_json(),
                source_post=post,
            )
            ev.compute_dedupe_key()
            results.append(ev)

        return results

    # ── private ─────────────────────────────────────────────────────────────

    def _build_user_message(self, posts: list[RawPost]) -> str:
        payload = {
            "posts": [
                {
                    "input_id": input_id,
                    "posted_date": p.taken_at[:10],
                    "text": p.combined_text(),
                }
                for input_id, p in enumerate(posts)
            ]
        }
        return json.dumps(payload, ensure_ascii=False)

    def _call_api_with_retry(
        self,
        user_msg: str,
        max_retries: int = 4,
    ) -> _BatchResponse:
        delay = 2.0
        last_err: Exception | None = None

        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.parse(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    response_format=_BatchResponse,
                    temperature=0,
                )
                print(f"Input token: {response.usage}")
                parsed = response.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("Model refused to produce structured output.")
                return parsed
            except RateLimitError as e:
                last_err = e
                if attempt + 1 == max_retries:
                    break
                print(f"[extractor] API error (attempt {attempt+1}): {e}. Retrying in {delay}s…")
                time.sleep(delay)
                delay *= 2
            except APIStatusError as e:
                if e.status_code in (408, 409) or e.status_code >= 500:
                    last_err = e
                    if attempt + 1 == max_retries:
                        break
                    print(
                        f"[extractor] API error (attempt {attempt+1}): {e}. "
                        f"Retrying in {delay}s…"
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                if e.status_code in (401, 403, 404):
                    raise ExtractionUnavailableError(
                        f"OpenAI request cannot be processed: {e}"
                    ) from e
                raise RuntimeError(f"OpenAI extraction failed: {e}") from e
            except Exception as e:
                raise RuntimeError(f"OpenAI extraction failed: {e}") from e

        raise ExtractionUnavailableError(
            f"OpenAI extraction failed after {max_retries} retries: {last_err}"
        )

if __name__ == "__main__":
    from pprint import pprint

    if not config.OPENAI_API_KEY:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add it to .env or export it in your shell."
        )

    sample_posts = [
        RawPost(
            source="instagram",
            username="ubc_example",
            post_url="https://www.instagram.com/p/test/",
            taken_at="2026-06-30T12:00:00Z",
            caption=(
                "Join us for a board game night!\n"
                "July 10, 2026 at 6:30 PM\n"
                "AMS Nest, Room 2500\n"
                "Everyone is welcome."
            ),
            ocr_text=None,
        ),
        RawPost(
            source="instagram",
            username="ubc_example",
            post_url="https://www.instagram.com/p/test/",
            taken_at="2026-06-30T12:00:00Z",
            caption=(
                "Join us for a board game night!\n"
                "July 10, 2026 at 6:30 PM\n"
                "AMS Nest, Room 2500\n"
                "Everyone is welcome."
            ),
            ocr_text=None,
        ),
    ]

    extractor = OpenAIExtractor(
        api_key=config.OPENAI_API_KEY,
        model=config.OPENAI_MODEL,
    )

    for event in extractor.extract_batch(sample_posts):
        pprint(event.to_dict())
