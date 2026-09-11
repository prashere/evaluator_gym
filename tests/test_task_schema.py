import json
from pathlib import Path

import jsonschema


def test_schema_file_valid_json():
    schema_path = Path("tasks/task.schema.json")
    schema = json.loads(schema_path.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
