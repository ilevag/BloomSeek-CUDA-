"""
Quick SQLite inspector for address databases.
Usage (defaults to ETH DB):
    python _tmp_inspect.py [db_path] [table] [column]
"""
import binascii
import json
import pathlib
import sqlite3
import sys


def main():
    db_path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "databases/eth_addresses.db")
    table = sys.argv[2] if len(sys.argv) > 2 else "addresses"
    column = sys.argv[3] if len(sys.argv) > 3 else "address"

    print(f"DB: {db_path}")
    if not db_path.exists():
        print("Database file not found")
        return

    print(f"Size: {db_path.stat().st_size / (1024*1024):,.2f} MB")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.cursor()

    page_size = cur.execute("PRAGMA page_size").fetchone()[0]
    page_count = cur.execute("PRAGMA page_count").fetchone()[0]
    print(f"Page size: {page_size}, pages: {page_count}, approx MB: {page_size*page_count/(1024*1024):,.2f}")

    tables = cur.execute("SELECT name, type FROM sqlite_master WHERE type in ('table','view') ORDER BY name").fetchall()
    print("Tables:", tables)

    schemas = {name: cur.execute(f"PRAGMA table_info('{name}')").fetchall() for name, _ in tables}
    print("Schema:", json.dumps(schemas, ensure_ascii=False, indent=2))

    try:
        sample = cur.execute(f"SELECT {column}, typeof({column}), length({column}) FROM {table} LIMIT 5").fetchall()
        preview = []
        for val, typ, length in sample:
            prefix = binascii.hexlify(val[:8]).decode() if isinstance(val, (bytes, bytearray)) else str(val)[:40]
            preview.append({"type": typ, "len": length, "prefix": prefix})
        print("Sample:", json.dumps(preview, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Sample query failed: {e}")

    con.close()


if __name__ == "__main__":
    main()
