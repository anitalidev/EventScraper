"""
Shared data models used across all pipeline stages.
Adding a new ingestion source (Eventbrite, Discord, etc.) means building a
scraper that produces RawPost objects — everything downstream is source-agnostic.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class RawPost:
    """One post as it comes off the wire from any scraper."""
    source: str                  # e.g. "instagram"
    username: str
    post_url: str
    taken_at: str                # ISO-8601 UTC
    caption: str
    image_path: Optional[str] = None
    ocr_text: Optional[str] = None

    def combined_text(self) -> str:
        """Caption plus any OCR text, used as the AI input."""
        parts = [self.caption]
        if self.ocr_text:
            parts.append(f"[Image text: {self.ocr_text}]")
        return "\n".join(parts).strip()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtractedEvent:
    """
    Normalised event record produced by the AI extractor and validated before
    being written to storage.
    Status flow: review ↔ approved ↔ review, review ↔ rejected, approved → published (final).
    """
    is_event: bool
    confidence: float            # 0.0 – 1.0
    confidence_reason: str
    title: str
    date: Optional[str]          # YYYY-MM-DD or None
    time: Optional[str]          # HH:MM (24 h) or None
    location: Optional[str]
    description: str
    source_url: str
    organizer: str
    vibes: list[str]             # subset of UBC Discovery vibe taxonomy
    source_label: str = "manual"  # e.g. "instagram", "Newsletter", "manual"
    status: str = "review"       # published | review | rejected | approved
    dedupe_key: Optional[str] = None
    raw_ai_response: Optional[str] = None
    validation_errors: list[str] = field(default_factory=list)
    image_url: Optional[str] = None

    # ── raw post that produced this event (not written to DB) ──────────────
    source_post: Optional[RawPost] = field(default=None, repr=False)

    # ── allowed status transitions ─────────────────────────────────────────
    _TRANSITIONS: dict[str, set[str]] = {
        "review":   {"approved", "rejected"},
        "approved": {"review", "published"},
        "rejected": {"review"},
        "published": set(),
    }

    @classmethod
    def check_transition(cls, from_status: str, to_status: str) -> None:
        """Raise ValueError if the transition is not allowed."""
        allowed = cls._TRANSITIONS.get(from_status, set())
        if to_status not in allowed:
            raise ValueError(f"Cannot transition from {from_status!r} to {to_status!r}")

    def _transition(self, target: str) -> None:
        self.check_transition(self.status, target)
        self.status = target

    def approve(self) -> None:
        self._transition("approved")

    def reject(self) -> None:
        self._transition("rejected")

    def send_back_to_review(self) -> None:
        self._transition("review")

    def publish(self) -> None:
        self._transition("published")

    def compute_dedupe_key(self) -> None:
        parts = "|".join([
            (self.title or "").lower().strip(),
            (self.organizer or "").lower().strip(),
            (self.date or ""),
        ])
        self.dedupe_key = hashlib.sha1(parts.encode()).hexdigest()

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("source_post", None)
        return d
