"""create earnings_insights

Revision ID: fe2eb5a292a1
Revises: 4aee5f48ae19
Create Date: 2026-09-09 23:01:10.760357

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'fe2eb5a292a1'
down_revision: Union[str, Sequence[str], None] = '4aee5f48ae19'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'earnings_insights',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('transcript_id', sa.BigInteger(), nullable=False),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('guidance_direction', sa.String(), nullable=True),  # raised | maintained | lowered | none_given
        sa.Column('sentiment_score', sa.Numeric(), nullable=True),  # -1..1
        sa.Column('risks', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('qoq_changes', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('generated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['transcript_id'], ['earnings_transcripts.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    # One row per transcript - each extraction pass
    op.create_index('ix_earnings_insights_transcript_id', 'earnings_insights', ['transcript_id'], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_earnings_insights_transcript_id', table_name='earnings_insights')
    op.drop_table('earnings_insights')
