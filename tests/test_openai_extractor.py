import unittest
import json
from typing import get_type_hints

from extractors.openai_extractor import (
    OpenAIExtractor,
    _BatchResponse,
    _SingleEvent,
    _SYSTEM_PROMPT,
)
from models.event import ExtractedEvent, RawPost, Vibe


class OpenAIExtractorTest(unittest.TestCase):
    def test_schema_and_domain_model_share_vibe_type(self) -> None:
        self.assertEqual(get_type_hints(_SingleEvent)["vibes"], list[Vibe])
        self.assertEqual(get_type_hints(ExtractedEvent)["vibes"], list[Vibe])

    def test_source_url_and_organizer_come_from_raw_post(self) -> None:
        post = RawPost(
            source="instagram",
            username="ubc_club",
            post_url="https://www.instagram.com/p/example/",
            taken_at="2026-06-30T12:00:00Z",
            caption="Board game night on July 10 at 6:30 PM.",
        )
        response = _BatchResponse(
            results=[
                _SingleEvent(
                    input_id=0,
                    is_event=True,
                    confidence=0.9,
                    title="Board Game Night",
                    date="2026-07-10",
                    time="18:30",
                    location=None,
                    description="A board game night open to attendees.",
                    vibes=["social"],
                )
            ]
        )
        extractor = OpenAIExtractor.__new__(OpenAIExtractor)
        extractor._call_api_with_retry = lambda _: response

        event = extractor.extract_batch([post])[0]

        self.assertEqual(event.source_url, post.post_url)
        self.assertEqual(event.organizer, post.username)
        self.assertIs(event.source_post, post)

    def test_results_are_correlated_by_id_not_response_order(self) -> None:
        posts = [
            RawPost(
                source="instagram",
                username=f"club_{i}",
                post_url=f"https://example.com/{i}",
                taken_at="2026-06-30T12:00:00Z",
                caption=f"Event {i}",
            )
            for i in (1, 2)
        ]
        response = _BatchResponse(results=[
            self._single_event(1, "Second"),
            self._single_event(0, "First"),
        ])
        extractor = OpenAIExtractor.__new__(OpenAIExtractor)
        extractor._call_api_with_retry = lambda _: response

        events = extractor.extract_batch(posts)

        self.assertEqual(
            [(event.title, event.source_url) for event in events],
            [
                ("First", "https://example.com/1"),
                ("Second", "https://example.com/2"),
            ],
        )

    def test_missing_response_id_is_rejected(self) -> None:
        posts = [
            RawPost(
                source="instagram",
                username=f"club_{i}",
                post_url=f"https://example.com/{i}",
                taken_at="2026-06-30T12:00:00Z",
                caption=f"Event {i}",
            )
            for i in (1, 2)
        ]
        response = _BatchResponse(results=[
            self._single_event(0, "First"),
        ])
        extractor = OpenAIExtractor.__new__(OpenAIExtractor)
        extractor._call_api_with_retry = lambda _: response

        with self.assertRaisesRegex(ValueError, r"missing=\[1\]"):
            extractor.extract_batch(posts)

    def test_user_message_is_structured_json_with_input_ids(self) -> None:
        post = RawPost(
            source="instagram",
            username="club",
            post_url="https://example.com/1",
            taken_at="2026-06-30T12:00:00Z",
            caption="Text containing\\n---\\na delimiter",
        )
        extractor = OpenAIExtractor.__new__(OpenAIExtractor)

        payload = json.loads(extractor._build_user_message([post]))

        self.assertEqual(payload["posts"][0]["input_id"], 0)
        self.assertEqual(payload["posts"][0]["posted_date"], "2026-06-30")
        self.assertEqual(payload["posts"][0]["text"], post.caption)
        self.assertNotIn("username", payload["posts"][0])

    def test_prompt_uses_posted_date_but_leaves_recency_to_validation(self) -> None:
        self.assertIn("using that post's posted_date", _SYSTEM_PROMPT)
        self.assertIn("downstream validation handles recency", _SYSTEM_PROMPT)
        self.assertNotIn("upcoming event", _SYSTEM_PROMPT)

    @staticmethod
    def _single_event(input_id: int, title: str) -> _SingleEvent:
        return _SingleEvent(
            input_id=input_id,
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
