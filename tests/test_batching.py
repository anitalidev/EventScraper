import unittest

from pipeline.batching import extract_with_bisection, iter_bounded_batches


class BatchingTest(unittest.TestCase):
    def test_batches_respect_item_and_character_limits(self) -> None:
        items = ["aaa", "bbbb", "cc", "d"]

        batches = list(iter_bounded_batches(
            items,
            max_items=3,
            max_chars=6,
            size_of=len,
        ))

        self.assertEqual(batches, [["aaa"], ["bbbb", "cc"], ["d"]])

    def test_oversized_item_is_yielded_intact(self) -> None:
        batches = list(iter_bounded_batches(
            ["oversized"],
            max_items=2,
            max_chars=3,
            size_of=len,
        ))

        self.assertEqual(batches, [["oversized"]])

    def test_invalid_limits_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size"):
            list(iter_bounded_batches(
                ["item"],
                max_items=0,
                max_chars=10,
                size_of=len,
            ))
        with self.assertRaisesRegex(ValueError, "batch_max_input_chars"):
            list(iter_bounded_batches(
                ["item"],
                max_items=1,
                max_chars=0,
                size_of=len,
            ))

    def test_failed_batch_is_split_until_bad_item_is_isolated(self) -> None:
        def extract(items: list[str]) -> list[str]:
            if "bad" in items:
                raise ValueError("bad input")
            return [item.upper() for item in items]

        results, failures, calls = extract_with_bisection(
            ["one", "bad", "two"], extract
        )

        self.assertEqual(results, ["ONE", "TWO"])
        self.assertEqual([failure.items for failure in failures], [["bad"]])
        self.assertEqual(str(failures[0].error), "bad input")
        self.assertEqual(calls, 5)

    def test_request_wide_failure_is_not_split(self) -> None:
        def extract(items: list[str]) -> list[str]:
            raise RuntimeError("service unavailable")

        results, failures, calls = extract_with_bisection(
            ["one", "two"],
            extract,
            should_split=lambda _error: False,
        )

        self.assertEqual(results, [])
        self.assertEqual(failures[0].items, ["one", "two"])
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
