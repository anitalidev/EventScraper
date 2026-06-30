import json
import unittest

from extractors.email_extractor import (
    EmailExtractor,
    _BatchResponse,
    _EmailEvent,
    _EmailResult,
    _SYSTEM_PROMPT,
)
from models.event import RawPost


class EmailExtractorTest(unittest.TestCase):
    def test_results_are_correlated_and_returned_in_input_order(self) -> None:
        posts = [self._post(1), self._post(2)]
        response = _BatchResponse(results=[
            _EmailResult(
                input_id=1,
                events=[self._event("Second")],
            ),
            _EmailResult(
                input_id=0,
                events=[self._event("First")],
            ),
        ])
        extractor = EmailExtractor.__new__(EmailExtractor)
        extractor._call_api_with_retry = lambda _: response

        events = extractor.extract_batch(posts)

        self.assertEqual(
            [(event.title, event.source_url) for event in events],
            [
                ("First", "https://example.com/1"),
                ("Second", "https://example.com/2"),
            ],
        )

    def test_user_message_is_structured_json_with_input_ids(self) -> None:
        extractor = EmailExtractor.__new__(EmailExtractor)

        payload = json.loads(extractor._build_user_message([self._post(1)]))

        self.assertEqual(payload["emails"][0]["input_id"], 0)
        self.assertEqual(payload["emails"][0]["received_date"], "2026-06-30")
        self.assertNotIn("from", payload["emails"][0])
        self.assertNotIn("gmail_link", payload["emails"][0])

    def test_prompt_uses_received_date_but_leaves_recency_to_validation(self) -> None:
        self.assertIn("using that email's received_date", _SYSTEM_PROMPT)
        self.assertIn("downstream validation handles recency", _SYSTEM_PROMPT)
        self.assertNotIn("upcoming event", _SYSTEM_PROMPT)

    @staticmethod
    def _post(index: int) -> RawPost:
        return RawPost(
            source="email",
            username=f"sender_{index}",
            post_url=f"https://example.com/{index}",
            taken_at="2026-06-30T12:00:00Z",
            caption=f"Event {index}",
        )

    @staticmethod
    def _event(title: str) -> _EmailEvent:
        return _EmailEvent(
            is_event=True,
            confidence=0.9,
            title=title,
            date="2026-07-10",
            time="18:30",
            location=None,
            description="An event.",
            vibes=["social"],
        )


if __name__ == "__main__":
    unittest.main()
