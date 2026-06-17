import json


def to_ndjson(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False) + "\n"
