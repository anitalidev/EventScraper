"""
Email-specific extractor.

Key difference from OpenAIExtractor: a single email can contain multiple events
(e.g. a newsletter listing this week's talks). The response schema wraps each
email's results in its own list, so one email → zero or more ExtractedEvents.
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


# ── Response schema ──────────────────────────────────────────────────────────

class _EmailEvent(BaseModel):
    is_event: bool = Field(description=(
        "True only for a real event with a specific date. "
        "False for job postings, deadlines, or vague announcements."
    ))
    confidence: float = Field(ge=0.0, le=1.0, description=(
        "How confident you are this is an actual event with extractable details."
    ))
    title: str = Field(description="Short, descriptive event title.")
    date: Optional[str] = Field(default=None, description="ISO date YYYY-MM-DD if found.")
    time: Optional[str] = Field(default=None, description="Start time HH:MM (24 h) if found.")
    location: Optional[str] = Field(default=None, description="Venue or location string.")
    description: str = Field(description="1-2 sentence summary of what the event is.")
    vibes: list[Vibe] = Field(description="Best matching event categories.")


class _EmailResult(BaseModel):
    input_id: int = Field(description="The input_id copied exactly from the input email.")
    events: list[_EmailEvent] = Field(description=(
        "All events found in this email. Empty list if none."
    ))


class _BatchResponse(BaseModel):
    results: list[_EmailResult] = Field(description=(
        "One entry per email, in the same order as the input."
    ))


# ── Prompt ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an event-extraction assistant for a university event aggregator.

Your task is to analyse emails and extract ALL real, specific events from each
one. A single email (e.g. a newsletter or digest) may contain multiple events
— extract every one of them.

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

Resolve relative dates (such as "tomorrow" or "this Friday") and dates without
a year using that email's received_date. Extract qualifying events whether
their date is past, present, or future; downstream validation handles recency.

Assign confidence based on how many concrete details are present:
- 0.9+ : date, time, location, and clear title all present
- 0.7-0.9 : date present plus at least one of time / location
- 0.5-0.7 : only a date is inferable
- below 0.5 : very uncertain

You will receive a list of emails with input IDs. Return exactly one result for
each email. Copy its input_id exactly into the result; do not omit, duplicate,
or invent IDs. Each result has an "events" array (which may be empty)."""


# ── Extractor ────────────────────────────────────────────────────────────────

class EmailExtractor(BaseExtractor):

    def __init__(self, api_key: str, model: str = config.OPENAI_MODEL):
        # Retry here so retry accounting and request-wide failure handling stay
        # visible to the pipeline rather than stacking with SDK retries.
        self._client = OpenAI(api_key=api_key, max_retries=0)
        self._model = model

    def extract_batch(self, posts: list[RawPost]) -> list[ExtractedEvent]:
        if not posts:
            return []

        user_msg = self._build_user_message(posts)
        batch_response = self._call_api_with_retry(user_msg)

        posts_by_id = {
            input_id: post for input_id, post in enumerate(posts)
        }
        response_ids = [item.input_id for item in batch_response.results]
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
            item.input_id: item for item in batch_response.results
        }
        results: list[ExtractedEvent] = []
        for input_id, post in posts_by_id.items():
            email_result = response_by_id[input_id]
            for item in email_result.events:
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

    def _build_user_message(self, posts: list[RawPost]) -> str:
        payload = {
            "emails": [
                {
                    "input_id": input_id,
                    "received_date": p.taken_at[:10],
                    "text": p.caption,
                }
                for input_id, p in enumerate(posts)
            ]
        }
        return json.dumps(payload, ensure_ascii=False)

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
            except RateLimitError as e:
                last_err = e
                if attempt + 1 == max_retries:
                    break
                print(f"[email_extractor] API error (attempt {attempt+1}): {e}. Retrying in {delay}s…")
                time.sleep(delay)
                delay *= 2
            except APIStatusError as e:
                if e.status_code in (408, 409) or e.status_code >= 500:
                    last_err = e
                    if attempt + 1 == max_retries:
                        break
                    print(
                        f"[email_extractor] API error (attempt {attempt+1}): "
                        f"{e}. Retrying in {delay}s…"
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                if e.status_code in (401, 403, 404):
                    raise ExtractionUnavailableError(
                        f"OpenAI request cannot be processed: {e}"
                    ) from e
                raise RuntimeError(f"Email extraction failed: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Email extraction failed: {e}") from e

        raise ExtractionUnavailableError(
            f"Email extraction failed after {max_retries} retries: {last_err}"
        )
