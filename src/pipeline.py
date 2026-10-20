import sys

from llm_sdk import Small_LLM_Model

from src.decoder import (
    choose_function_name,
    generate_parameters,
)
from src.models import (
    FunctionCallResult,
    FunctionDefinition,
    PromptItem,
)
from src.vocab import TokenTable


def validate_parameters(
    function: FunctionDefinition,
    parameters: dict[str, object],
) -> None:
    """Validate generated arguments against the function schema."""
    expected_names = set(function.parameters)
    actual_names = set(parameters)

    if actual_names != expected_names:
        raise ValueError(
            "Parameter names do not match the function definition."
        )

    for name, definition in function.parameters.items():
        value = parameters[name]
        parameter_type = definition.type.lower()

        if parameter_type == "string":
            valid = isinstance(value, str)
        
        elif parameter_type in ("number", "float"):
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            )

        elif parameter_type == "integer":
            valid = (
                isinstance(value, int)
                and not isinstance(value, bool)
            )
        elif parameter_type == "boolean":
            valid = isinstance(value, bool)
        else:
            raise ValueError(
                f"Unsupported parameter type: {definition.type}"
            )

        if not valid:
            raise ValueError(
                f"Invalid type for parameter '{name}': "
                f"expected {parameter_type}, "
                f"got {type(value).__name__}."
            )


def process_prompts(
    model: Small_LLM_Model,
    functions: list[FunctionDefinition],
    prompts: list[PromptItem],
    table: TokenTable,
) -> list[FunctionCallResult]:
    """Process every input prompt."""
    by_name = {
        function.name: function
        for function in functions
    }

    results: list[FunctionCallResult] = []
    failures = 0

    for item in prompts:
        try:
            function_name = choose_function_name(
                model,
                item.prompt,
                functions,
            )

            function = by_name[function_name]

            parameters = generate_parameters(
                model,
                item.prompt,
                function,
                table,
            )

            validate_parameters(function, parameters)

            result = FunctionCallResult(
                prompt=item.prompt,
                name=function_name,
                parameters=parameters,
            )

            results.append(result)

            print(
                f"{item.prompt} -> "
                f"{function_name} {parameters}"
            )

        except Exception as error:
            failures += 1

            print(
                f"Error processing "
                f"'{item.prompt}': {error}",
                file=sys.stderr,
            )

    if failures:
        raise RuntimeError(
            f"{failures} prompt(s) failed. "
            "Output file was not written."
        )

    return results
