def make_strict_schema(schema: dict) -> dict:
    """Make a JSON schema compatible with OpenAI strict mode.

    Strict mode requires every object node to have:
    - additionalProperties: false
    - required listing every key in properties (nullable fields use anyOf with null)
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object" or "properties" in schema:
            schema["additionalProperties"] = False
            if "properties" in schema:
                schema["required"] = list(schema["properties"].keys())
        for value in schema.values():
            if isinstance(value, dict):
                make_strict_schema(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        make_strict_schema(item)
    return schema
