"""Tiny JSON-schema subset validator (no dependencies).

Supports: type, properties, required, additionalProperties, enum, minLength, maxLength,
minimum, maximum, items. Enough for tool argument schemas.
"""

import math
import re

_TYPES = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


def validate(value, schema, path="args"):
    """Return a list of error strings (empty when valid)."""
    errors = []
    t = schema.get("type")
    if t is not None and t not in _TYPES:
        return [f"{path}: schema error, unknown type {t!r}"]
    if t is not None and not _TYPES[t](value):
        return [f"{path}: expected {t}, got {type(value).__name__}"]
    if isinstance(value, float) and not math.isfinite(value):
        return [f"{path}: NaN/Infinity not allowed"]
    if "enum" in schema and not any(
        type(value) is type(e) and value == e for e in schema["enum"]
    ):
        errors.append(f"{path}: must be one of {schema['enum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if len(value) > schema.get("maxLength", len(value)):
            errors.append(f"{path}: longer than {schema['maxLength']}")
    if (
        isinstance(value, str)
        and "pattern" in schema
        and re.fullmatch(schema["pattern"], value) is None
    ):
        errors.append(f"{path}: does not match required format")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum {schema['maximum']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required '{key}'")
        for key, sub in value.items():
            if key in props:
                errors += validate(sub, props[key], f"{path}.{key}")
            elif schema.get("additionalProperties", True) is False:
                errors.append(f"{path}: unexpected property '{key}'")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors += validate(item, schema["items"], f"{path}[{i}]")
    return errors
