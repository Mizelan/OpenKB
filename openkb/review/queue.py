"""ReviewQueue — persist review items to .openkb/review_queue.json."""
from __future__ import annotations

import json
from pathlib import Path

from openkb.review.models import ReviewItem


class ReviewQueue:
    """Persistent queue of review items stored in .openkb/review_queue.json."""

    def __init__(self, openkb_dir: Path) -> None:
        self._dir = openkb_dir
        self._path = openkb_dir / "review_queue.json"
        self._items: list[ReviewItem] = []
        self._load()

    def _load(self) -> None:
        """Load items from disk (empty list if file missing)."""
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._items = [ReviewItem.from_dict(d) for d in data]
            except (json.JSONDecodeError, KeyError, ValueError):
                self._items = []
        else:
            self._items = []

    def save(self) -> None:
        """Persist current items to disk."""
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps([item.to_dict() for item in self._items], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list(self) -> list[ReviewItem]:
        """Return current pending items in insertion order."""
        return list(self._items)

    def add(self, items: list[ReviewItem]) -> None:
        """Append items and save."""
        self._items.extend(items)
        self.save()

    def accept(self, index: int) -> ReviewItem:
        """Remove and return item at *index*, then save. Raises IndexError if out of range."""
        item = self._items.pop(index)
        self.save()
        return item

    def skip(self, index: int) -> None:
        """Remove item at *index* without returning it. Raises IndexError if out of range."""
        self._items.pop(index)
        self.save()