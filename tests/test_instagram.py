import json
import unittest
from datetime import date, datetime, timezone
from unittest.mock import patch

from scrapers.instagram import fetch_posts


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _item(index: int) -> dict:
    return {
        "pk": str(index),
        "code": f"post-{index}",
        "taken_at": int(
            datetime(2026, 6, 30, tzinfo=timezone.utc).timestamp()
        ),
        "caption": {"text": f"Post {index}"},
    }


class InstagramPostLimitTest(unittest.TestCase):
    @patch("scrapers.instagram._download_image")
    @patch("scrapers.instagram.urllib.request.urlopen")
    def test_image_is_keyed_by_account_and_post_id(
        self,
        urlopen,
        download_image,
    ) -> None:
        item = _item(123)
        item["image_versions2"] = {
            "candidates": [{"url": "https://cdn.example/post.jpg"}]
        }
        urlopen.return_value = _Response({
            "items": [item],
            "more_available": False,
        })
        download_image.return_value = "/tmp/ig_example_123.jpg"

        posts, error = fetch_posts(
            "example",
            date(2026, 6, 1),
            date(2026, 6, 30),
        )

        self.assertIsNone(error)
        self.assertEqual(posts[0].image_path, "/tmp/ig_example_123.jpg")
        download_image.assert_called_once_with(
            "https://cdn.example/post.jpg",
            "example",
            "123",
        )

    @patch("scrapers.instagram.urllib.request.urlopen")
    def test_default_limit_fetches_one_page_of_twelve(self, urlopen) -> None:
        urlopen.return_value = _Response({
            "items": [_item(i) for i in range(12)],
            "next_max_id": "next",
            "more_available": True,
        })

        posts, error = fetch_posts(
            "example",
            date(2026, 6, 1),
            date(2026, 6, 30),
            download_images=False,
        )

        self.assertIsNone(error)
        self.assertEqual(len(posts), 12)
        self.assertEqual(urlopen.call_count, 1)
        self.assertIn("count=12", urlopen.call_args.args[0].full_url)

    @patch("scrapers.instagram.urllib.request.urlopen")
    def test_limit_can_span_pages_without_overfetching(self, urlopen) -> None:
        urlopen.side_effect = [
            _Response({
                "items": [_item(i) for i in range(12)],
                "next_max_id": "next",
                "more_available": True,
            }),
            _Response({
                "items": [_item(i) for i in range(12, 24)],
                "more_available": False,
            }),
        ]

        posts, error = fetch_posts(
            "example",
            date(2026, 6, 1),
            date(2026, 6, 30),
            download_images=False,
            post_limit=15,
        )

        self.assertIsNone(error)
        self.assertEqual(len(posts), 15)
        self.assertEqual(urlopen.call_count, 2)
        self.assertIn("count=3", urlopen.call_args.args[0].full_url)


if __name__ == "__main__":
    unittest.main()
