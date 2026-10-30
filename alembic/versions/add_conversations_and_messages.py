
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d71a9c42e8f0"
down_revision: Union[str, Sequence[str], None] = "c90c30034a75"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_conversations_id"),
        "conversations",
        ["id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_conversations_document_id"),
        "conversations",
        ["document_id"],
        unique=False,
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("sources", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_messages_id"),
        "messages",
        ["id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_messages_conversation_id"),
        "messages",
        ["conversation_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_messages_conversation_id"),
        table_name="messages",
    )
    op.drop_index(
        op.f("ix_messages_id"),
        table_name="messages",
    )
    op.drop_table("messages")

    op.drop_index(
        op.f("ix_conversations_document_id"),
        table_name="conversations",
    )
    op.drop_index(
        op.f("ix_conversations_id"),
        table_name="conversations",
    )
    op.drop_table("conversations")
