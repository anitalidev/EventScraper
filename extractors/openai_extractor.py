"""
OpenAI-backed extractor using structured JSON outputs (response_format).

The extractor treats scraped content as anonymous text+URL pairs — there is no
concept of a "post" here. OpenAI receives a numbered list of content items and
returns a flat list of events across all of them. One item can yield zero,
one, or many events.
"""
from __future__ import annotations
import time
from typing import Optional

from openai import OpenAI, RateLimitError, APIStatusError
from pydantic import BaseModel, Field

import config
from extractors.base import BaseExtractor
from models.event import RawPost, ExtractedEvent


# ── Pydantic schema that OpenAI will enforce ────────────────────────────────

class _Event(BaseModel):
    title: str = Field(description="Short, descriptive event title.")
    date: Optional[str] = Field(default=None, description="ISO date YYYY-MM-DD if found.")
    time: Optional[str] = Field(default=None, description="Start time HH:MM (24 h) if found.")
    location: Optional[str] = Field(default=None, description="Venue or location string.")
    description: str = Field(description="1-2 sentence summary of what the event is.")
    source_url: str = Field(description="The URL of the content item this event was found in.")
    organizer: str = Field(description="The @username or name of the account that posted.")
    vibes: list[str] = Field(description=(
        "One or more vibes that best describe this event. "
        "Choose only from: social, career, academic, arts, culture, "
        "outdoors, sports, food, wellness, volunteering."
    ))


class _BatchResponse(BaseModel):
    events: list[_Event]


# ── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an event-extraction assistant for a university event aggregator.

You will receive a numbered list of content items scraped from social media.
Each item has a source URL and text content. Extract ALL real, specific,
upcoming events you find across all items.

Return a flat list of events under the "events" key — do NOT group by content
item. A single item may yield multiple events (e.g. a series of workshops on
different dates); extract each as a separate entry. If an item contains no
real events, produce nothing for it.

Only extract events that:
- Have (or imply) a specific date
- Are open to attendees

Do NOT extract:
- Job postings or hiring announcements
- Application or deadline reminders
- Executive or committee recruitment
- General club promotions with no specific date
- Recurring programme announcements with no specific occurrence

For each event, set source_url to the URL of the content item it came from.

For vibes, pick one or more from this fixed list: social, career, academic,
arts, culture, outdoors, sports, food, wellness, volunteering.

"""


class OpenAIExtractor(BaseExtractor):

    def __init__(self, api_key: str, model: str = config.OPENAI_MODEL):
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def extract_batch(self, posts: list[RawPost]) -> list[ExtractedEvent]:
        if not posts:
            return []

        user_msg = self._build_user_message(posts)
        raw_response = self._call_api_with_retry(user_msg)

        results: list[ExtractedEvent] = []
        for item in raw_response.events:
            ev = ExtractedEvent(
                is_event=True,
                title=item.title,
                date=item.date,
                time=item.time,
                location=item.location,
                description=item.description,
                source_url=item.source_url,
                organizer=item.organizer,
                vibes=item.vibes,
                raw_ai_response=item.model_dump_json(),
            )
            ev.compute_dedupe_key()
            results.append(ev)

        return results

    def _build_user_message(self, posts: list[RawPost]) -> str:
        lines = []
        for i, p in enumerate(posts, 1):
            lines.append(
                f"[{i}] URL: {p.post_url}\n"
                f"Content: {p.combined_text()}"
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
                print(f"[extractor] API error (attempt {attempt+1}): {e}. Retrying in {delay}s…")
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                raise RuntimeError(f"OpenAI extraction failed: {e}") from e

        raise RuntimeError(f"OpenAI extraction failed after {max_retries} retries: {last_err}")
