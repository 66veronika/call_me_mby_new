import json

import numpy as np
from llm_sdk import Small_LLM_Model

from src.models import FunctionDefinition
from src.vocab import TokenTable
import math

DIGITS = "0123456789"


def encode_text(
    model: Small_LLM_Model,
    text: str,
) -> list[int]:
    """Encode text into a flat list of token IDs."""
    encoded = model.encode(text)
    token_ids = encoded.squeeze(0).tolist()

    if isinstance(token_ids, int):
        return [token_ids]

    return [int(token_id) for token_id in token_ids]


def choose_candidate(
    model: Small_LLM_Model,
    context_ids: list[int],
    candidates: dict[str, list[int]],
) -> str:
    """Let the LLM choose one candidate token path."""
    if not candidates:
        raise ValueError("No candidates were provided.")

    active = candidates.copy()
    generated: list[int] = []

    while active:
        if len(active) == 1:
            return next(iter(active))

        position = len(generated)

        allowed = {
            sequence[position]
            for sequence in active.values()
            if position < len(sequence)
        }

        if not allowed:
            raise RuntimeError("No valid candidate token remains.")

        if len(allowed) == 1:
            next_token = next(iter(allowed))
        else:
            logits = model.get_logits_from_input_ids(
                context_ids + generated
            )
            next_token = max(
                allowed,
                key=lambda token_id: logits[token_id],
            )

        generated.append(next_token)

        active = {
            name: sequence
            for name, sequence in active.items()
            if sequence[:len(generated)] == generated
        }

    raise RuntimeError("Could not choose a candidate.")


def build_function_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Build the prompt used for function selection."""
    lines = [
        "Choose the best function for the user request.",
        "Return only the function name.",
        "",
        f"User request: {user_prompt}",
        "",
        "Available functions:",
    ]

    for function in functions:
        parameters = ", ".join(
            f"{name}: {definition.type}"
            for name, definition in function.parameters.items()
        )
        lines.append(
            f"- {function.name}({parameters}): {function.description}"
        )

    lines.extend(["", "Function:"])

    return "\n".join(lines)


def choose_function_name(
    model: Small_LLM_Model,
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Choose the function using constrained token paths."""
    if not functions:
        raise ValueError("No functions were provided.")

    prompt = build_function_prompt(user_prompt, functions)
    context_ids = encode_text(model, prompt)

    candidates = {
        function.name: encode_text(model, f" {function.name}\n")
        for function in functions
    }

    return choose_candidate(model, context_ids, candidates)


def normalized_number(text: str) -> str:
    """Remove permitted leading whitespace from a number."""
    return text.lstrip(" ")


def is_numeric_prefix(text: str, *, integer: bool = False) -> bool:
    """Return whether text can still become a JSON number (or integer)."""
    value = normalized_number(text)

    if value == "":
        return True

    index = 0
    length = len(value)

    if value[index] == "-":
        index += 1
        if index == length:
            return True

    if value[index] == "0":
        index += 1
        if index < length and value[index] in DIGITS:
            return False
    elif value[index] in "123456789":
        index += 1
        while index < length and value[index] in DIGITS:
            index += 1
    else:
        return False

    if integer or index == length:
        return index == length

    if value[index] == ".":
        index += 1
        if index == length:
            return True
        if value[index] not in DIGITS:
            return False
        while index < length and value[index] in DIGITS:
            index += 1
        if index == length:
            return True

    if value[index] in "eE":
        index += 1
        if index == length:
            return True
        if value[index] in "+-":
            index += 1
            if index == length:
                return True
        if value[index] not in DIGITS:
            return False
        while index < length and value[index] in DIGITS:
            index += 1

    return index == length


def is_complete_numeric(text: str, *, integer: bool = False) -> bool:
    """Return whether text is a complete JSON number (or integer)."""
    value = normalized_number(text)
    return (
        value != ""
        and value[-1] in DIGITS
        and is_numeric_prefix(value, integer=integer)
    )


def parse_numeric(
    text: str,
    *,
    integer: bool,
) -> int | float:
    """Convert a complete numeric string safely."""
    value = text.lstrip(" ")

    if integer:
        return int(value)

    number = float(value)

    if not math.isfinite(number):
        raise ValueError("Generated number is not finite.")

    return number


def best_stop_score(
    logits: list[float],
    table: TokenTable,
) -> float:
    """Return the highest score for a JSON delimiter token."""
    valid_ids = [
        token_id
        for token_id in table.stop_ids
        if token_id < len(logits)
    ]

    if not valid_ids:
        return float("-inf")

    return max(logits[token_id] for token_id in valid_ids)


def generate_numeric(
    model: Small_LLM_Model,
    prompt_ids: list[int],
    table: TokenTable,
    *,
    integer: bool = False,
    max_tokens: int = 16,
) -> int | float:
    """Generate one JSON number or integer."""
    generated: list[int] = []
    text = ""
    allowed_ids = table.integer_ids if integer else table.number_ids
    kind = "integer" if integer else "number"

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(prompt_ids + generated)

        best = max(
            (
                i for i in allowed_ids
                if i < len(logits)
                and is_numeric_prefix(text + table.text[i], integer=integer)
            ),
            key=lambda i: logits[i],
            default=None,
        )

        if is_complete_numeric(text, integer=integer):
            if best is None or best_stop_score(logits, table) > logits[best]:
                return parse_numeric(text, integer=integer)

        if best is None:
            raise RuntimeError(f"No valid {kind} token is available.")

        generated.append(best)
        text += table.text[best]

    if is_complete_numeric(text, integer=integer):
        return parse_numeric(text, integer=integer)

    raise RuntimeError(f"{kind.capitalize()} generation did not finish.")


def scan_string_body(text: str) -> tuple[bool, int | None]:
    """Validate a JSON string body and locate its closing quote.

    Incomplete but valid prefixes are accepted so generation can continue.
    """
    index = 0
    hex_digits = "0123456789abcdefABCDEF"
    simple_escapes = '"\\/bfnrt'

    while index < len(text):
        char = text[index]

        if char == '"':
            return True, index

        if ord(char) < 32:
            return False, None

        if char != "\\":
            index += 1
            continue

        # A trailing backslash is a valid incomplete prefix.
        if index + 1 >= len(text):
            return True, None

        escape = text[index + 1]

        if escape in simple_escapes:
            index += 2
            continue

        if escape != "u":
            return False, None

        # A Unicode escape requires exactly four hexadecimal digits.
        hex_part = text[index + 2:index + 6]

        if any(char not in hex_digits for char in hex_part):
            return False, None

        if len(hex_part) < 4:
            return True, None

        index += 6

    return True, None


def generate_string(
    model: Small_LLM_Model,
    prompt_ids: list[int],
    table: TokenTable,
    max_tokens: int = 128,
) -> str:
    """Generate a JSON string body.

    The opening quote must already be present in the prompt.
    """
    generated: list[int] = []

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(prompt_ids + generated)
        ranked = np.argsort(np.asarray(logits))[::-1]

        selected: int | None = None

        for raw_token_id in ranked:
            token_id = int(raw_token_id)

            if token_id >= len(table.text):
                continue

            candidate = model.decode(generated + [token_id])

            if "\ufffd" in candidate:
                continue

            valid, closing_quote = scan_string_body(candidate)

            if not valid:
                continue

            if closing_quote is not None:
                body = candidate[:closing_quote]

                try:
                    value = json.loads(f'"{body}"')
                except json.JSONDecodeError:
                    continue

                return str(value)

            selected = token_id
            break

        if selected is None:
            raise RuntimeError("No valid string token is available.")

        generated.append(selected)

    raise RuntimeError("String generation did not finish.")


def generate_boolean(
    model: Small_LLM_Model,
    prompt_ids: list[int],
) -> bool:
    """Generate a JSON boolean."""
    candidates = {
        "true": encode_text(model, "true"),
        "false": encode_text(model, "false"),
    }

    result = choose_candidate(model, prompt_ids, candidates)

    return result == "true"


def build_parameters_prompt(
    user_prompt: str,
    function: FunctionDefinition,
) -> str:
    """Build the argument-generation prompt."""
    schema = ", ".join(
        f"{name}: {definition.type}"
        for name, definition in function.parameters.items()
    )

    return (
        "Extract input arguments for a function call.\n"
        "Do not execute the function.\n\n"
        "Examples:\n"
        "Request: Combine 7 and 11\n"
        "Function: fn_pair(left: number, right: number)\n"
        'JSON arguments: {"left": 7, "right": 11}\n\n'
        "Request: Echo the string 'orange'\n"
        "Function: fn_echo(text: string)\n"
        'JSON arguments: {"text": "orange"}\n\n'
        "Request: Replace digits in 'A12B' with X\n"
        "Function: fn_replace(source: string, "
        "pattern: string, replacement: string)\n"
        'JSON arguments: {"source": "A12B", '
        '"pattern": "\\\\d+", "replacement": "X"}\n\n'
        f"Request: {user_prompt}\n"
        f"Function: {function.name}({schema})\n"
        f"Description: {function.description}\n"
        "JSON arguments: "
    )


def generate_parameters(
    model: Small_LLM_Model,
    user_prompt: str,
    function: FunctionDefinition,
    table: TokenTable,
) -> dict[str, object]:
    """Generate all required parameters in one JSON context."""
    base_prompt = build_parameters_prompt(user_prompt, function)

    parameters: dict[str, object] = {}
    json_prefix = "{"

    for index, (name, definition) in enumerate(
        function.parameters.items()
    ):
        if index > 0:
            json_prefix += ", "

        json_prefix += f"{json.dumps(name)}: "

        parameter_type = definition.type.lower()
        context = base_prompt + json_prefix

        value: object
        if parameter_type == "string":
            value = generate_string(
                model, encode_text(model, context + '"'), table
            )

        elif parameter_type in ("number", "float"):
            value = generate_numeric(
                model, encode_text(model, context), table
            )

        elif parameter_type == "integer":
            value = generate_numeric(
                model, encode_text(model, context), table, integer=True
            )
        elif parameter_type == "boolean":
            value = generate_boolean(model, encode_text(model, context))
        else:
            raise ValueError(
                f"Unsupported parameter type: {definition.type}"
            )

        parameters[name] = value
        json_prefix += json.dumps(value, ensure_ascii=False)

    return parameters
