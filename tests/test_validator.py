import unittest
from datetime import date

from models.event import ExtractedEvent
from validation.validator import validate


class ValidatorTest(unittest.TestCase):
    def test_past_event_is_rejected(self) -> None:
        event = self._event("2026-06-29")

        validate(event, current_date=date(2026, 6, 30))

        self.assertEqual(event.status, "rejected")
        self.assertIn(
            "event date is in the past: 2026-06-29",
            event.validation_errors,
        )

    def test_event_today_is_not_rejected_as_past(self) -> None:
        event = self._event("2026-06-30")

        validate(event, current_date=date(2026, 6, 30))

        self.assertEqual(event.status, "review")
        self.assertNotIn("event date is in the past", event.validation_errors)

    def test_future_event_is_not_rejected_as_past(self) -> None:
        event = self._event("2026-07-01")

        validate(event, current_date=date(2026, 6, 30))

        self.assertEqual(event.status, "review")
        self.assertNotIn("event date is in the past", event.validation_errors)

    @staticmethod
    def _event(event_date: str) -> ExtractedEvent:
        return ExtractedEvent(
            is_event=True,
            confidence=0.9,
            title="Test Event",
            date=event_date,
            time=None,
            location=None,
            description="A test event.",
            source_url="https://example.com/event",
            organizer="Test Organizer",
            vibes=["social"],
        )


if __name__ == "__main__":
    unittest.main()
