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
from extractors.base import BaseExtractor
from models.event import RawPost, ExtractedEvent

# ── Pydantic schema that OpenAI will enforce ────────────────────────────────

class _SingleEvent(BaseModel):
    is_event: bool = Field(description=(
        "True only for real upcoming events with a specific date. "
        "False for job postings, application deadlines, exec recruitment, "
        "general ads, or vague announcements."
    ))
    confidence: float = Field(ge=0.0, le=1.0, description=(
        "How confident you are this is an actual event with extractable details."
    ))
    confidence_reason: str = Field(description=(
        "One sentence explaining the confidence score."
    ))
    title: str = Field(description="Short, descriptive event title.")
    date: Optional[str] = Field(default=None, description="ISO date YYYY-MM-DD if found.")
    time: Optional[str] = Field(default=None, description="Start time HH:MM (24 h) if found.")
    location: Optional[str] = Field(default=None, description="Venue or location string.")
    description: str = Field(description="1-2 sentence summary of what the event is.")
    source_url: str = Field(description="The Instagram post URL provided.")
    organizer: str = Field(description="The @username of the account that posted.")
    event_type: str = Field(description=(
        "One of: Workshop, Networking, Career, Social, Academic, Sports, "
        "Wellness, Volunteer, Arts, Culture, Food, Other."
    ))


class _BatchResponse(BaseModel):
    results: list[_SingleEvent]


# ── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an event-extraction assistant for a university event aggregator.

Your task is to analyse Instagram post captions and decide whether each post
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

You will receive a numbered list of posts. Return a JSON object with a
"results" array of exactly the same length, in the same order.
Each element must conform to the schema exactly."""


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
                confidence_reason=item.confidence_reason,
                title=item.title,
                date=item.date,
                time=item.time,
                location=item.location,
                description=item.description,
                source_url=item.source_url or (post.post_url if post else ""),
                organizer=item.organizer,
                event_type=item.event_type,
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
                f"URL: {p.post_url}\n"
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
                response = self._client.beta.chat.completions.parse(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    response_format=_BatchResponse,
                    temperature=0,
                )
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
