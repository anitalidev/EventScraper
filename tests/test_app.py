import unittest
from unittest.mock import patch

from app import app


class ScrapeRouteTest(unittest.TestCase):
    def test_post_limit_is_passed_to_pipeline(self) -> None:
        client = app.test_client()
        with patch("app.run", return_value={}) as run:
            response = client.post(
                "/scrape",
                json={
                    "start_date": "2026-06-01",
                    "end_date": "2026-06-30",
                    "channels": "example",
                    "api_key": "test",
                    "post_limit": 24,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(run.call_args.kwargs["post_limit"], 24)

    def test_invalid_post_limits_are_rejected_before_pipeline_runs(self) -> None:
        client = app.test_client()
        request_body = {
            "start_date": "2026-06-01",
            "end_date": "2026-06-30",
            "channels": "example",
            "api_key": "test",
        }

        with patch("app.run") as run:
            for value in (0, -1, "abc"):
                with self.subTest(post_limit=value):
                    response = client.post(
                        "/scrape",
                        json={**request_body, "post_limit": value},
                    )
                    self.assertEqual(response.status_code, 400)

        run.assert_not_called()

    def test_invalid_batch_sizes_are_rejected_before_pipeline_runs(self) -> None:
        client = app.test_client()
        request_body = {
            "start_date": "2026-06-01",
            "end_date": "2026-06-30",
            "channels": "example",
            "api_key": "test",
        }

        with patch("app.run") as run:
            for value in (0, -1, 51, "abc"):
                with self.subTest(batch_size=value):
                    response = client.post(
                        "/scrape",
                        json={**request_body, "batch_size": value},
                    )
                    self.assertEqual(response.status_code, 400)

        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
