"""enable pgvector extension

Revision ID: c902a842d2a8
Revises: 5ff12f56060e
Create Date: 2026-08-19 16:02:13.601350

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c902a842d2a8'
down_revision: Union[str, Sequence[str], None] = '5ff12f56060e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Enables `vector(N)` columns and similarity search (<->, <#>, <=> operators).
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP EXTENSION IF EXISTS vector")
