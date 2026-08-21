"""create memo_embeddings table

Revision ID: 5477c6e8f7e8
Revises: c902a842d2a8
Create Date: 2026-08-20 20:47:46.968830

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '5477c6e8f7e8'
down_revision: Union[str, Sequence[str], None] = 'c902a842d2a8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Dimension is hardcoded (not imported from src.embeddings.service) so this
# migration stays a frozen, self-contained record regardless of future
# application-code changes - see src/embeddings/service.py for why 1536.
_EMBEDDING_DIMENSION = 1536


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'memo_embeddings',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('ticker', sa.String(), nullable=False),
        sa.Column('scan_run_id', sa.BigInteger(), nullable=False),
        sa.Column('memo_text', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(_EMBEDDING_DIMENSION), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['scan_run_id'], ['scan_runs.id']),
        sa.ForeignKeyConstraint(['ticker'], ['companies.ticker']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_memo_embeddings_ticker_scan_run_id', 'memo_embeddings', ['ticker', 'scan_run_id'], unique=True,
    )
    # HNSW (not IVFFlat) so it's usable immediately without a data-dependent
    # `lists` parameter to tune; cosine ops to match .cosine_distance() in
    # src/analytics/memo_search.py.
    op.execute(
        "CREATE INDEX ix_memo_embeddings_embedding_hnsw ON memo_embeddings "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS ix_memo_embeddings_embedding_hnsw")
    op.drop_index('ix_memo_embeddings_ticker_scan_run_id', table_name='memo_embeddings')
    op.drop_table('memo_embeddings')
