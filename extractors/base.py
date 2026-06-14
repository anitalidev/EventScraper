"""
Abstract extractor interface.

To swap AI providers, subclass BaseExtractor and pass the new implementation
to the pipeline.  The rest of the codebase only depends on this interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from models.event import RawPost, ExtractedEvent


class BaseExtractor(ABC):

    @abstractmethod
    def extract_batch(self, posts: list[RawPost]) -> list[ExtractedEvent]:
        """
        Given a batch of raw posts, return one ExtractedEvent per post.
        Posts that are not events should still be returned with is_event=False
        so the caller can account for every input position.
        """
        ...
