from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine


def _table_exists_sqlite(conn, table_name: str) -> bool:
    result = conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='table' AND name=:name"),
        {"name": table_name},
    ).first()
    return result is not None


def _column_names_sqlite(conn, table_name: str) -> set[str]:
    rows = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
    return {str(row[1]) for row in rows}


def _add_column_sqlite_if_missing(conn, table_name: str, column: str, definition: str) -> None:
    columns = _column_names_sqlite(conn, table_name)
    if column not in columns:
        conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column} {definition}"))


def bootstrap_auth_schema(engine: Engine) -> None:
    if engine.dialect.name != "sqlite":
        return

    with engine.begin() as conn:
        if not _table_exists_sqlite(conn, "users"):
            return

        _add_column_sqlite_if_missing(conn, "users", "tenant_id", "VARCHAR(64)")
        _add_column_sqlite_if_missing(conn, "users", "microsoft_oid", "VARCHAR(64)")
        _add_column_sqlite_if_missing(conn, "users", "role", "VARCHAR(20) NOT NULL DEFAULT 'user'")
        _add_column_sqlite_if_missing(
            conn, "users", "auth_provider", "VARCHAR(20) NOT NULL DEFAULT 'microsoft'"
        )
        _add_column_sqlite_if_missing(conn, "users", "is_active", "BOOLEAN NOT NULL DEFAULT 1")
        _add_column_sqlite_if_missing(conn, "users", "token_budget", "INTEGER NOT NULL DEFAULT 0")

        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_users_tenant_id ON users (tenant_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_users_microsoft_oid ON users (microsoft_oid)"))
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_users_tenant_oid "
                "ON users (tenant_id, microsoft_oid)"
            )
        )

        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS user_permissions ("
                "id VARCHAR(36) PRIMARY KEY, "
                "user_id VARCHAR(36) NOT NULL, "
                "permission_code VARCHAR(100) NOT NULL, "
                "FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE"
                ")"
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_user_permission "
                "ON user_permissions (user_id, permission_code)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_user_permissions_user_code "
                "ON user_permissions (user_id, permission_code)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS budget_requests ("
                "id VARCHAR(36) PRIMARY KEY, "
                "user_id VARCHAR(36) NOT NULL, "
                "requested_tokens INTEGER NOT NULL, "
                "note TEXT, "
                "status VARCHAR(20) NOT NULL DEFAULT 'pending', "
                "admin_note TEXT, "
                "decided_by_user_id VARCHAR(36), "
                "decided_at DATETIME, "
                "created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE, "
                "FOREIGN KEY(decided_by_user_id) REFERENCES users(id) ON DELETE SET NULL"
                ")"
            )
        )
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_budget_requests_user_id ON budget_requests (user_id)")
        )
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_budget_requests_status ON budget_requests (status)")
        )


def bootstrap_document_schema(engine: Engine) -> None:
    if engine.dialect.name != "sqlite":
        return

    with engine.begin() as conn:
        if not _table_exists_sqlite(conn, "document"):
            return

        _add_column_sqlite_if_missing(conn, "document", "storage_path", "VARCHAR(500)")
        _add_column_sqlite_if_missing(conn, "document", "content_type", "VARCHAR(100)")
        _add_column_sqlite_if_missing(
            conn, "document", "size_bytes", "INTEGER NOT NULL DEFAULT 0"
        )
        _add_column_sqlite_if_missing(
            conn, "document", "status", "VARCHAR(20) NOT NULL DEFAULT 'ready'"
        )
        _add_column_sqlite_if_missing(
            conn, "document", "chunk_count", "INTEGER NOT NULL DEFAULT 0"
        )
        _add_column_sqlite_if_missing(
            conn, "document", "created_at", "DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP"
        )
        _add_column_sqlite_if_missing(
            conn, "document", "updated_at", "DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP"
        )
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_document_owner_id ON document (owner_id)"))


def bootstrap_rag_retrieval_event_schema(engine: Engine) -> None:
    if engine.dialect.name != "sqlite":
        return

    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS rag_retrieval_events ("
                "id VARCHAR(36) PRIMARY KEY, "
                "message_id VARCHAR(36) NOT NULL, "
                "conversation_id VARCHAR(36) NOT NULL, "
                "user_id VARCHAR(36) NOT NULL, "
                "chat_mode VARCHAR(30) NOT NULL, "
                "route VARCHAR(30) NOT NULL, "
                "route_confidence VARCHAR(20) NOT NULL, "
                "original_query TEXT NOT NULL, "
                "rewritten_query TEXT NOT NULL, "
                "rewrite_used BOOLEAN NOT NULL DEFAULT 0, "
                "rewrite_source VARCHAR(20) NOT NULL DEFAULT 'none', "
                "rewrite_confidence VARCHAR(20) NOT NULL DEFAULT 'low', "
                "initial_k INTEGER NOT NULL DEFAULT 0, "
                "final_k INTEGER NOT NULL DEFAULT 0, "
                "retrieval_confidence VARCHAR(20) NOT NULL DEFAULT 'low', "
                "retrieved_chunk_count INTEGER NOT NULL DEFAULT 0, "
                "selected_chunk_count INTEGER NOT NULL DEFAULT 0, "
                "retrieved_document_ids JSON NOT NULL DEFAULT '[]', "
                "selected_context JSON NOT NULL DEFAULT '[]', "
                "fallback_reason VARCHAR(100), "
                "latency_ms INTEGER NOT NULL DEFAULT 0, "
                "created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE, "
                "FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE, "
                "FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE"
                ")"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_rag_retrieval_events_message_id "
                "ON rag_retrieval_events (message_id)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_rag_retrieval_events_conversation_id "
                "ON rag_retrieval_events (conversation_id)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_rag_retrieval_events_user_id "
                "ON rag_retrieval_events (user_id)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_rag_retrieval_events_conversation_created "
                "ON rag_retrieval_events (conversation_id, created_at)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_rag_retrieval_events_user_created "
                "ON rag_retrieval_events (user_id, created_at)"
            )
        )
