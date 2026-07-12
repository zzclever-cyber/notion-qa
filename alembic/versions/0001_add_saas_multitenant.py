"""add SaaS multi-tenant tables (users / knowledge_bases / documents)
and multi-tenant columns on chat_records

Revision ID: 0001_saas_mt
Revises:
Create Date: 2026-07-12

说明
----
本项目启动时通过 init_db() 的 create_all 建表，create_all 只会创建「缺失的表」，
不会给已存在的 chat_records 补列。因此本迁移做的是「增量」：
1. 新建 users / knowledge_bases / documents 三张表
2. 给 chat_records 补 user_id / kb_id / total_tokens 三列

在已有开发库上执行：  alembic upgrade head
"""
from alembic import op
import sqlalchemy as sa


revision = "0001_saas_mt"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---- users ----
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), server_default=""),
        sa.Column("display_name", sa.String(length=128), server_default=""),
        sa.Column("avatar_url", sa.String(length=512), server_default=""),
        sa.Column("github_login", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_github_login", "users", ["github_login"])

    # ---- knowledge_bases ----
    op.create_table(
        "knowledge_bases",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), server_default=""),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_knowledge_bases_user_id", "knowledge_bases", ["user_id"])

    # ---- documents ----
    op.create_table(
        "documents",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("kb_id", sa.String(length=36), sa.ForeignKey("knowledge_bases.id"), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("file_type", sa.String(length=16), server_default=""),
        sa.Column("chunk_count", sa.Integer(), server_default="0"),
        sa.Column("file_size_bytes", sa.Integer(), server_default="0"),
        sa.Column("status", sa.String(length=16), server_default="processing"),
        sa.Column("error_message", sa.Text(), server_default=""),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_documents_kb_id", "documents", ["kb_id"])

    # ---- chat_records: 新增多租户列 ----
    op.add_column("chat_records", sa.Column("user_id", sa.String(length=36), nullable=True))
    op.add_column("chat_records", sa.Column("kb_id", sa.String(length=36), nullable=True))
    op.add_column("chat_records", sa.Column("total_tokens", sa.Integer(), server_default="0"))
    op.create_index("ix_chat_records_user_id", "chat_records", ["user_id"])
    op.create_index("ix_chat_records_kb_id", "chat_records", ["kb_id"])


def downgrade() -> None:
    op.drop_index("ix_chat_records_kb_id", table_name="chat_records")
    op.drop_index("ix_chat_records_user_id", table_name="chat_records")
    op.drop_column("chat_records", "total_tokens")
    op.drop_column("chat_records", "kb_id")
    op.drop_column("chat_records", "user_id")

    op.drop_index("ix_documents_kb_id", table_name="documents")
    op.drop_table("documents")
    op.drop_index("ix_knowledge_bases_user_id", table_name="knowledge_bases")
    op.drop_table("knowledge_bases")
    op.drop_index("ix_users_github_login", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
