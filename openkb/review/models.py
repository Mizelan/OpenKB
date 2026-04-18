"""ReviewItem dataclass for the 2-step ingest pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field

VALID_REVIEW_TYPES = {"contradiction", "duplicate", "missing_page", "confirm", "suggestion"}


@dataclass
class ReviewItem:
    """A single review item produced by the analysis step."""

    type: str
    title: str
    description: str
    source_path: str
    affected_pages: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    options: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.type not in VALID_REVIEW_TYPES:
            raise ValueError(
                f"Invalid review type: {self.type!r}. "
                f"Must be one of: {', '.join(sorted(VALID_REVIEW_TYPES))}"
            )

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "source_path": self.source_path,
            "affected_pages": self.affected_pages,
            "search_queries": self.search_queries,
            "options": self.options,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ReviewItem:
        return cls(
            type=d["type"],
            title=d["title"],
            description=d["description"],
            source_path=d["source_path"],
            affected_pages=d.get("affected_pages", []),
            search_queries=d.get("search_queries", []),
            options=d.get("options", []),
        )