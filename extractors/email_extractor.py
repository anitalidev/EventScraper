"""
Email-specific extractor.

Key difference from OpenAIExtractor: a single email can contain multiple events
(e.g. a newsletter listing this week's talks). The response schema wraps each
email's results in its own list, so one email → zero or more ExtractedEvents.
"""
from __future__ import annotations

import time
from typing import Optional

from openai import OpenAI, RateLimitError, APIStatusError
from pydantic import BaseModel, Field

import config
from extractors.base import BaseExtractor
from models.event import RawPost, ExtractedEvent


# ── Response schema ──────────────────────────────────────────────────────────

class _EmailEvent(BaseModel):
    is_event: bool = Field(description=(
        "True only for a real upcoming event with a specific date. "
        "False for job postings, deadlines, or vague announcements."
    ))
    confidence: float = Field(ge=0.0, le=1.0, description=(
        "How confident you are this is an actual event with extractable details."
    ))
    confidence_reason: str = Field(description="One sentence explaining the confidence score.")
    title: str = Field(description="Short, descriptive event title.")
    date: Optional[str] = Field(default=None, description="ISO date YYYY-MM-DD if found.")
    time: Optional[str] = Field(default=None, description="Start time HH:MM (24 h) if found.")
    location: Optional[str] = Field(default=None, description="Venue or location string.")
    description: str = Field(description="1-2 sentence summary of what the event is.")
    source_url: str = Field(description="The Gmail link provided for this email.")
    organizer: str = Field(description="Sender's name or organisation, not their email address.")
    vibes: list[str] = Field(description=(
        "One or more vibes: social, career, academic, arts, culture, "
        "outdoors, sports, food, wellness, volunteering."
    ))


class _EmailResult(BaseModel):
    events: list[_EmailEvent] = Field(description=(
        "All events found in this email. Empty list if none."
    ))


class _BatchResponse(BaseModel):
    results: list[_EmailResult] = Field(description=(
        "One entry per email, in the same order as the input."
    ))


# ── Prompt ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an event-extraction assistant for a university event aggregator.

Your task is to analyse emails and extract ALL real, specific, upcoming events
from each one. A single email (e.g. a newsletter or digest) may contain
multiple events — extract every one of them.

For each event found, set is_event=true. If an email contains no events,
return an empty events list for it.

SKIP (do not extract) the following:
- Job postings or hiring announcements
- Applications open / deadline reminders
- Receipts, confirmations, or transactional emails
- General club promotions without a specific date

EXTRACT when the email describes an event that:
- Has a specific date (even if approximate)
- Is open to attendees
- Is a single occurrence or a specific series instance

For the organizer field, use the sender's name or organisation (not their email address).
For source_url, use the Gmail link provided for that email.
For vibes, pick one or more from: social, career, academic, arts, culture,
outdoors, sports, food, wellness, volunteering. Do not invent new values.

Assign confidence based on how many concrete details are present:
- 0.9+ : date, time, location, and clear title all present
- 0.7-0.9 : date present plus at least one of time / location
- 0.5-0.7 : only a date is inferable
- below 0.5 : very uncertain

You will receive a numbered list of emails. Return a JSON object with a
"results" array of exactly the same length, in the same order. Each element
has an "events" array (may be empty)."""


# ── Extractor ────────────────────────────────────────────────────────────────

class EmailExtractor(BaseExtractor):

    def __init__(self, api_key: str, model: str = config.OPENAI_MODEL):
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def extract_batch(self, posts: list[RawPost]) -> list[ExtractedEvent]:
        if not posts:
            return []

        user_msg = self._build_user_message(posts)
        batch_response = self._call_api_with_retry(user_msg)

        results: list[ExtractedEvent] = []
        for i, email_result in enumerate(batch_response.results):
            post = posts[i] if i < len(posts) else None
            for item in email_result.events:
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
                    organizer=post.username if post else item.organizer,
                    vibes=item.vibes,
                    raw_ai_response=item.model_dump_json(),
                    source_post=post,
                )
                ev.compute_dedupe_key()
                results.append(ev)

        return results

    def _build_user_message(self, posts: list[RawPost]) -> str:
        lines = []
        for i, p in enumerate(posts, 1):
            lines.append(
                f"[{i}] From: {p.username}  |  received: {p.taken_at[:10]}\n"
                f"Gmail link: {p.post_url}\n"
                f"{p.caption}"
            )
        return "\n\n---\n\n".join(lines)

    def _call_api_with_retry(self, user_msg: str, max_retries: int = 4) -> _BatchResponse:
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
                print(f"[email_extractor] API error (attempt {attempt+1}): {e}. Retrying in {delay}s…")
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                raise RuntimeError(f"Email extraction failed: {e}") from e

        raise RuntimeError(f"Email extraction failed after {max_retries} retries: {last_err}")
