"""Parse ---REVIEW--- blocks from LLM output."""
from __future__ import annotations

import logging

from openkb.json_utils import extract_json
from openkb.review.models import ReviewItem

logger = logging.getLogger(__name__)


def _extract_json_array(text: str) -> list:
    """Extract the first JSON array from text."""
    result = extract_json(text, start_char="[")
    return result if isinstance(result, list) else []


def parse_review_blocks(text: str) -> list[ReviewItem]:
    """Extract review items from ---REVIEW--- delimited blocks in LLM output.

    Each block after the delimiter should contain a JSON array of review item
    objects. Malformed JSON is logged and skipped.
    """
    items: list[ReviewItem] = []
    delimiter = "---REVIEW---"

    idx = text.find(delimiter)
    while idx != -1:
        body_start = idx + len(delimiter)
        next_delim = text.find(delimiter, body_start)
        body = text[body_start:next_delim] if next_delim != -1 else text[body_start:]
        body = body.strip()

        if body:
            raw_items = _extract_json_array(body)
            if raw_items:
                for raw in raw_items:
                    try:
                        items.append(ReviewItem.from_dict(raw))
                    except (KeyError, ValueError) as exc:
                        logger.warning("Skipping invalid review item: %s", exc)
            elif body.find("[") != -1:
                logger.warning("Failed to parse review block JSON in: %s", body[:80])

        idx = text.find(delimiter, body_start) if next_delim != -1 else -1

    return items