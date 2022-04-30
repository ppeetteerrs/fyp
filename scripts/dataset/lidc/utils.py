"""
LIDC dataset utility functions
"""

from typing import Any


def parse_subject(path: str, *_: Any) -> str:
    return path.split("/")[-4].replace("LIDC-IDRI-", "")
