"""ReviewItem dataclass for the 2-step ingest pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VALID_REVIEW_TYPES = {"contradiction", "duplicate", "missing_page", "confirm", "suggestion"}
VALID_ACTION_TYPES = {"alias_concept", "create_placeholder", "mark_stale"}
VALID_REVIEW_STATUSES = {"pending", "accepted", "applied", "skipped"}


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
    action_type: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"

    def __post_init__(self) -> None:
        if self.type not in VALID_REVIEW_TYPES:
            raise ValueError(
                f"Invalid review type: {self.type!r}. "
                f"Must be one of: {', '.join(sorted(VALID_REVIEW_TYPES))}"
            )
        if self.action_type is not None and self.action_type not in VALID_ACTION_TYPES:
            raise ValueError(
                f"Invalid action type: {self.action_type!r}. "
                f"Must be one of: {', '.join(sorted(VALID_ACTION_TYPES))}"
            )
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be a dictionary")
        if self.status not in VALID_REVIEW_STATUSES:
            raise ValueError(
                f"Invalid review status: {self.status!r}. "
                f"Must be one of: {', '.join(sorted(VALID_REVIEW_STATUSES))}"
            )

    def to_dict(self) -> dict:
        data = {
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "source_path": self.source_path,
            "affected_pages": self.affected_pages,
            "search_queries": self.search_queries,
            "options": self.options,
        }
        if self.action_type is not None:
            data["action_type"] = self.action_type
            data["payload"] = self.payload
            data["status"] = self.status
        elif self.payload:
            data["payload"] = self.payload
            data["status"] = self.status
        elif self.status != "pending":
            data["status"] = self.status
        return data

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
            action_type=d.get("action_type"),
            payload=d.get("payload", {}),
            status=d.get("status", "pending"),
        )
