import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from integrations.ubc_discovery import _build_payload, publish_event


def _event() -> dict:
    return {
        "id": 42,
        "title": "Campus Event",
        "description": "Description",
        "organizer": "example_club",
        "source_url": "https://www.instagram.com/p/example/",
        "vibes": '["social"]',
        "location": "The Nest",
        "date": "2026-07-01",
        "time": "18:00",
        "image_url": None,
    }


class UBCDiscoveryImageUploadTest(unittest.TestCase):
    def test_create_payload_matches_documented_schema(self) -> None:
        payload = _build_payload(_event())

        self.assertNotIn("event_picture_key", payload)
        self.assertNotIn("external_ref", payload)
        self.assertEqual(payload["event_date"], "2026-07-02T01:00:00+00:00")

    def test_bc_winter_dates_keep_the_permanent_utc_minus_seven_offset(self) -> None:
        event = _event()
        event["date"] = "2026-12-01"

        payload = _build_payload(event)

        self.assertEqual(payload["event_date"], "2026-12-02T01:00:00+00:00")

    @patch("integrations.ubc_discovery.find_raw_post_image")
    @patch("integrations.ubc_discovery.requests.post")
    @patch("integrations.ubc_discovery.config.UBC_DISCOVERY_API_KEY", "secret")
    @patch("integrations.ubc_discovery.config.UBC_DISCOVERY_API_URL", "http://api")
    def test_publishes_then_uploads_raw_instagram_image(
        self,
        post: Mock,
        find_raw_post_image: Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "ig_example_club_123.jpg")
            Image.new("RGB", (8, 8), "red").save(image_path)
            find_raw_post_image.return_value = image_path

            create_response = Mock(
                ok=True,
                status_code=200,
                content=b"{}",
                text="",
            )
            create_response.json.return_value = {
                "id": "event123",
                "title": "Campus Event",
                "created_at": "2026-06-30T00:00:00Z",
            }
            presign_response = Mock(ok=True, status_code=200, text="")
            presign_response.json.return_value = {
                "upload_url": "https://s3.example/upload",
                "fields": {
                    "key": "event-pictures/event123.webp",
                    "Content-Type": "image/webp",
                },
                "file_key": "event-pictures/event123.webp",
                "max_file_size_bytes": 1_000_000,
            }
            upload_response = Mock()
            upload_response.raise_for_status.return_value = None
            post.side_effect = [
                create_response,
                presign_response,
                upload_response,
            ]

            created = publish_event(_event())

            self.assertEqual(created.ubc_event_id, "event123")
            self.assertEqual(post.call_args_list[0].args[0], "http://api/events")
            self.assertEqual(
                post.call_args_list[1].args[0],
                "http://api/events/event123/presigned-upload",
            )
            upload_call = post.call_args_list[2]
            self.assertEqual(upload_call.args[0], "https://s3.example/upload")
            filename, uploaded, content_type = upload_call.kwargs["files"]["file"]
            self.assertEqual(filename, "ig_example_club_123.webp")
            self.assertEqual(content_type, "image/webp")
            self.assertTrue(uploaded.read().startswith(b"RIFF"))
            self.assertTrue(os.path.exists(image_path))


if __name__ == "__main__":
    unittest.main()
