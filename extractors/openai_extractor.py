"""
OpenAI-backed extractor using structured JSON outputs (response_format).

Structured outputs guarantee the model returns valid JSON that matches our
Pydantic schema, eliminating the need for manual parsing or error-prone regex
post-processing.

Retry strategy: exponential back-off on rate-limit / server errors (tenacity).
"""
from __future__ import annotations
import time
from typing import Optional

from openai import OpenAI, RateLimitError, APIStatusError
from pydantic import BaseModel, Field

import config
from extractors.base import BaseExtractor
from models.event import ExtractedEvent, RawPost, Vibe

# ── Pydantic schema that OpenAI will enforce ────────────────────────────────

class _SingleEvent(BaseModel):
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
describes a real, specific, upcoming event.

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

Assign confidence based on how many concrete details are present:
- 0.9+ : date, time, location, and clear title all present
- 0.7-0.9 : date present plus at least one of time / location
- 0.5-0.7 : only a date is inferable
- below 0.5 : very uncertain — likely not a real event

Return exactly one result for each input post, preserving the input order.
"""


class OpenAIExtractor(BaseExtractor):
    """
    Extracts events from batches of RawPost objects using the OpenAI API
    with structured JSON output mode.
    """

    def __init__(self, api_key: str, model: str = config.OPENAI_MODEL):
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def extract_batch(self, posts: list[RawPost]) -> list[ExtractedEvent]:
        if not posts:
            return []

        user_msg = self._build_user_message(posts)
        raw_response = self._call_api_with_retry(user_msg)

        results: list[ExtractedEvent] = []
        for i, item in enumerate(raw_response.results):
            post = posts[i] if i < len(posts) else None
            ev = ExtractedEvent(
                is_event=item.is_event,
                confidence=item.confidence,
                title=item.title,
                date=item.date,
                time=item.time,
                location=item.location,
                description=item.description,
                source_url=post.post_url if post else "",
                organizer=post.username if post else "",
                vibes=item.vibes,
                raw_ai_response=item.model_dump_json(),
                source_post=post,
            )
            ev.compute_dedupe_key()
            results.append(ev)

        return results

    # ── private ─────────────────────────────────────────────────────────────

    def _build_user_message(self, posts: list[RawPost]) -> str:
        lines = []
        for i, p in enumerate(posts, 1):
            lines.append(
                f"[{i}] @{p.username}  |  posted: {p.taken_at[:10]}\n"
                f"Text: {p.combined_text()}"
            )
        return "\n\n---\n\n".join(lines)

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
            except (RateLimitError, APIStatusError) as e:
                last_err = e
                print(f"[extractor] API error (attempt {attempt+1}): {e}. Retrying in {delay}s…")
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                raise RuntimeError(f"OpenAI extraction failed: {e}") from e

        raise RuntimeError(f"OpenAI extraction failed after {max_retries} retries: {last_err}")

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
