"""create earnings_transcripts, earnings_chunks

Revision ID: 4aee5f48ae19
Revises: 5477c6e8f7e8
Create Date: 2026-09-03 12:24:30.663376

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '4aee5f48ae19'
down_revision: Union[str, Sequence[str], None] = '5477c6e8f7e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Hardcoded (not imported from src.embeddings.service), same as
# 5477c6e8f7e8's memo_embeddings.embedding - this migration stays a frozen,
# self-contained record regardless of future application-code changes. Must
# match memo_embeddings.embedding's dimension: both are voyage-large-2
# vectors (src/embeddings/service.py) sharing the same retrieval pipeline
# (architecture.md §6), so a mismatch here would silently break nothing
# structurally but make the two tables incompatible with a single shared
# query/model dimension expectation.
_EMBEDDING_DIMENSION = 1536


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'earnings_transcripts',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('ticker', sa.String(), nullable=False),
        sa.Column('fiscal_quarter', sa.String(), nullable=False),
        sa.Column('fiscal_year', sa.Integer(), nullable=False),
        sa.Column('raw_text', sa.Text(), nullable=False),
        sa.Column('source_url', sa.String(), nullable=True),
        sa.Column('ingested_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['ticker'], ['companies.ticker']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_earnings_transcripts_ticker_fiscal_year_fiscal_quarter',
        'earnings_transcripts', ['ticker', 'fiscal_year', 'fiscal_quarter'], unique=True,
    )

    op.create_table(
        'earnings_chunks',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('transcript_id', sa.BigInteger(), nullable=False),
        sa.Column('chunk_type', sa.String(), nullable=False),  # prepared_remarks | qna
        sa.Column('chunk_text', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(_EMBEDDING_DIMENSION), nullable=False),
        sa.ForeignKeyConstraint(['transcript_id'], ['earnings_transcripts.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_earnings_chunks_transcript_id', 'earnings_chunks', ['transcript_id'])
    # HNSW (not IVFFlat), cosine ops - same rationale/index type as
    # ix_memo_embeddings_embedding_hnsw (5477c6e8f7e8): usable immediately
    # with no data-dependent `lists` parameter to tune, and cosine ops match
    # the .cosine_distance() search pattern architecture.md §6 specifies for
    # earnings_chunks once ingestion lands.
    op.execute(
        "CREATE INDEX ix_earnings_chunks_embedding_hnsw ON earnings_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS ix_earnings_chunks_embedding_hnsw")
    op.drop_index('ix_earnings_chunks_transcript_id', table_name='earnings_chunks')
    op.drop_table('earnings_chunks')
    op.drop_index('ix_earnings_transcripts_ticker_fiscal_year_fiscal_quarter', table_name='earnings_transcripts')
    op.drop_table('earnings_transcripts')
