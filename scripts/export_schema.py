"""Regenerate schemas/request.schema.json from the Request model.

Usage: uv run python scripts/export_schema.py
"""

from pathlib import Path

from cubesip.request import write_request_json_schema

target = Path(__file__).resolve().parents[1] / "schemas" / "request.schema.json"
write_request_json_schema(target)
print(f"wrote {target}")
