import sqlite3
from typing import Any, Dict, List, Tuple, Optional
import os


class SQLiteTool:
    """
    Simple wrapper around SQLite for:
    - connecting to the DB
    - introspecting schema (tables, columns)
    - executing SQL queries safely
    """

    def __init__(self, db_path: str = "data/northwind.sqlite") -> None:
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"SQLite DB not found at: {db_path}")
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None => autocommit mode
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # access columns by name
        return conn

    def get_tables(self) -> List[str]:
        """
        Return a list of table names in the database.
        """
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table'
                ORDER BY name;
                """
            )
            return [row["name"] for row in cur.fetchall()]
        finally:
            conn.close()

    def get_schema(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Return a mapping:
            table_name -> list of (column_name, data_type)
        """
        conn = self._connect()
        schema: Dict[str, List[Tuple[str, str]]] = {}
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table'
                ORDER BY name;
                """
            )
            tables = [row["name"] for row in cur.fetchall()]

            for t in tables:
                cur.execute(f"PRAGMA table_info('{t}')")
                cols = [(row["name"], row["type"]) for row in cur.fetchall()]
                schema[t] = cols
        finally:
            conn.close()

        return schema

    def execute_sql(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
        max_rows: int = 1000,
    ) -> Dict[str, Any]:
        """
        Execute a SELECT-like SQL query and return:
        {
            "ok": bool,
            "error": str or None,
            "columns": [col1, col2, ...],
            "rows": [ [v11, v12, ...], [v21, v22, ...], ... ]
        }
        """
        conn = self._connect()
        result: Dict[str, Any] = {
            "ok": False,
            "error": None,
            "columns": [],
            "rows": [],
        }

        try:
            cur = conn.cursor()
            if params is None:
                cur.execute(sql)
            else:
                cur.execute(sql, params)

            # If it's a SELECT, fetch the rows
            if cur.description is not None:
                columns = [desc[0] for desc in cur.description]
                rows_raw = cur.fetchmany(max_rows)
                rows = [list(r) for r in rows_raw]

                result["ok"] = True
                result["columns"] = columns
                result["rows"] = rows
            else:
                # Non-SELECT (UPDATE/INSERT/etc.), just commit
                conn.commit()
                result["ok"] = True
        except Exception as e:
            result["ok"] = False
            result["error"] = str(e)
        finally:
            conn.close()

        return result
