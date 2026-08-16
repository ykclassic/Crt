"""Backward-compatible database initialization wrapper.

The old implementation dropped and recreated the signals table. That violated
pipeline database ownership and could destroy signal history. Initialization
now delegates to the non-destructive pipeline persistence layer.
"""

from config import DB_FILE
from db_utils import init_db


def initialize_database() -> None:
    """Initialize or safely migrate the pipeline-owned signal database."""
    init_db(DB_FILE)


if __name__ == "__main__":
    initialize_database()
