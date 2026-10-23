*This project has been created as part of the 42 curriculum by vskopova*

# Call Me Maybe

## Description

**Call Me Maybe** is a project about function calling in Large Language Models (LLMs). Its goal is to transform natural-language requests into structured function calls containing a function name and its required parameters.

Instead of answering a question directly, the system identifies the most appropriate function, extracts its arguments, and generates structured values that can be validated and processed by another program.

For example, given the prompt:

> What is the sum of 40 and 2?

The expected result is:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 40,
    "b": 2
  }
}
```

The project uses the `Qwen/Qwen3-0.6B` language model through the provided `llm_sdk` package. It implements constrained decoding to guide token selection and generate values that follow the expected parameter types.

### Main features

- Select a function using an LLM rather than hardcoded keyword rules.
- Extract function arguments from natural-language requests.
- Support string, number, integer, and boolean parameters.
- Use constrained token selection for function names and argument generation.
- Validate function definitions and input prompts using Pydantic.
- Read input data from JSON files and write results to a JSON output file.
- Handle invalid input files and report errors during processing.

## Instructions

### Requirements

- Python 3.10 or later.
- [uv](https://docs.astral.sh/uv/) for dependency management.
- Access to the Hugging Face model repository to download the model and tokenizer on the first run.
- Sufficient memory and processing resources to load and run the model.

The project configuration specifies Python 3.13, while the package metadata requires Python 3.10 or later. Use a Python version compatible with the dependencies and the provided SDK.

### Installation

Clone the repository and enter the project directory.

Install the dependencies:

```bash
uv sync
```

The project includes the `llm_sdk` package as a local workspace dependency.

### Running the project

Run the program with the default input files:

```bash
uv run python -m src
```

Alternatively, use the Makefile:

```bash
make run
```

The program uses the following default paths:

- Function definitions: `data/input/functions_definition.json`
- Input prompts: `data/input/function_calling_tests.json`
- Output results: `data/output/function_calling_results.json`

You can specify custom paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Development commands

```bash
make install
make debug
make lint
make lint-strict
make clean
```

- `install`: Synchronizes dependencies using uv.
- `debug`: Starts the program with Python's debugger.
- `lint`: Runs flake8 and mypy with the configured checks.
- `lint-strict`: Runs flake8 and strict mypy checks.
- `clean`: Removes Python caches and generated output files.

## Algorithm Explanation

The implementation uses constrained decoding to control which tokens the language model can select while generating function names and parameter values.

### 1. Function selection

The system builds a prompt containing the user's request and the available function definitions, including their names, parameters, and descriptions.

The request is tokenized, and each available function name is encoded as a candidate token sequence.

The `choose_candidate` function compares candidate sequences using the model's logits. When multiple candidates remain possible, it selects the highest-scoring allowed next token and eliminates candidates that do not match the generated prefix.

This process continues until a function has been selected.

### 2. Vocabulary preparation

The `build_token_table` function loads the model vocabulary and decodes its tokens.

It builds a reusable table containing:

- Decoded text for each token ID.
- Candidate tokens suitable for numeric values.
- Candidate tokens suitable for integer values.
- Tokens that may represent JSON value delimiters or stopping points.

Preparing this table before processing prompts avoids rebuilding these token groups for every argument.

### 3. Parameter generation

Once a function has been selected, the system builds an argument-generation prompt containing the request, function description, and parameter schema.

The `generate_parameters` function processes each parameter according to its declared type:

- **String:** Generates a JSON string body while checking valid escaping and string structure.
- **Number:** Generates a numeric value while checking whether each candidate token maintains a valid number.
- **Integer:** Applies additional restrictions so that the generated value remains an integer.
- **Boolean:** Selects between the valid JSON values `true` and `false`.

For numeric generation, invalid token candidates are excluded. The decoder also compares candidate continuation scores with JSON delimiter scores to decide when a complete value should end.

For strings, the decoder checks escape sequences, control characters, and closing quotes before accepting a generated value.

### 4. Validation

After generating the parameters, `validate_parameters` checks that:

- All expected parameter names are present.
- No unexpected parameter names are included.
- Each value has the correct Python type.
- Boolean values are not incorrectly accepted as integers or numbers.

Pydantic models validate the input function definitions and prompts before inference begins.

### 5. Result generation

The pipeline combines the original prompt, selected function name, and generated parameters into a result object.

The results are serialized to JSON in the configured output file. If any prompt fails during processing, the pipeline reports the failure and does not write a partial results file.

## Design Decisions

### Using an LLM for function selection

The project uses the language model to select among the available functions instead of relying on hardcoded keyword matching. This allows the selection process to use the descriptions of the available functions.

### Constrained decoding

Small language models may generate invalid structured output. Restricting candidate tokens according to the expected structure reduces this risk compared with relying on prompting alone.

### Type-specific generation

Strings, numbers, integers, and booleans have different structural requirements. Separate generation strategies make it possible to enforce the appropriate constraints for each supported type.

### Pydantic validation

Pydantic models provide a consistent way to validate input definitions, prompts, and result objects. Additional parameter validation checks that generated values match their declared types.

### Separation of responsibilities

The implementation separates its main responsibilities into modules:

- `src/decoder.py`: Function selection and constrained argument generation.
- `src/vocab.py`: Vocabulary loading and token-table preparation.
- `src/models.py`: Pydantic data models.
- `src/io_utils.py`: JSON input and output.
- `src/pipeline.py`: Prompt processing and parameter validation.
- `src/__main__.py`: Command-line interface and application entry point.
- `llm_sdk/`: The provided interface for local language-model inference.

## Performance Analysis

The project aims to improve the structural reliability of function calling while using a relatively small language model.

Constrained decoding limits the tokens available during generation, while vocabulary preprocessing allows token information to be reused across prompts.

However, constrained decoding does not guarantee that the model understands the request correctly or extracts the intended arguments. A syntactically valid value can still be semantically incorrect.

Performance should be evaluated using the following measures:

- **Function-selection accuracy:** Percentage of requests assigned to the correct function.
- **Argument accuracy:** Percentage of generated arguments matching the expected values.
- **Type correctness:** Percentage of arguments matching the declared parameter types.
- **JSON validity:** Whether the generated output can be parsed as JSON and follows the required structure.
- **Execution time:** Total time needed to process the complete test set.

The subject targets at least 90% correct function selection and argument extraction, 100% valid structured output, and execution of the test prompts within five minutes on standard hardware.

These are target requirements, not measured results. Actual performance must be established by running the implementation against the provided and additional test cases.

## Challenges Faced

The implementation addresses several challenges associated with constrained generation:

- **Choosing among function names:** Candidate token sequences may share prefixes, so the decoder must retain all compatible candidates until the choice becomes unambiguous.
- **Generating valid numbers:** Numeric generation must reject invalid continuations, including malformed decimal and exponent notation.
- **Handling JSON strings:** Escaped quotes, backslashes, Unicode escapes, and control characters require careful validation.
- **Maintaining correct types:** Generated values must match the parameter types declared in the function definitions.
- **Handling errors:** Missing files, invalid JSON, unsupported parameter types, and failed prompt processing must be reported clearly.

Further testing is needed to determine how reliably these cases are handled across different models, prompts, and function definitions.

## Testing Strategy

The provided test file contains prompts for addition, greetings, string reversal, square roots, regular-expression substitutions, and related tasks.

The implementation should be tested at several levels:

1. **Input validation:** Test missing files, malformed JSON, and invalid function definitions.
2. **Function selection:** Check that each request selects the intended function.
3. **Argument extraction:** Compare generated parameter values with expected values.
4. **Type validation:** Test strings, numbers, integers, booleans, and unsupported types.
5. **Edge cases:** Include empty strings, negative numbers, decimals, large values, escaped characters, and ambiguous requests.
6. **Output validation:** Parse the output file as JSON and verify the required keys and parameter types.
7. **Performance:** Measure total runtime and accuracy across the complete test set.
8. **Static analysis:** Run `make lint` and, if possible, `make lint-strict`.

No test or benchmark results are claimed here until these checks have actually been executed.

## Example Usage

### Input prompt

```json
[
  {
    "prompt": "What is the sum of 2 and 3?"
  },
  {
    "prompt": "Greet john"
  },
  {
    "prompt": "Reverse the string 'hello'"
  }
]
```

### Example output

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  },
  {
    "prompt": "Greet john",
    "name": "fn_greet",
    "parameters": {
      "name": "john"
    }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {
      "s": "hello"
    }
  }
]
```

This is an illustrative example of the expected format, not a verified output from a program run. The generated result contains the function call and its arguments; it does not execute the selected function.

## Resources


https://docshield.tungstenautomation.com/RPA/en_us/10.6.0_p2wddr4n2j/help/kap_help/designstudio/c_jsonterminology.html

https://learn.microsoft.com/en-us/dotnet/ai/conceptual/understanding-tokens
- **JSON specification and Python support:** https://docs.python.org/3/library/json.html
- **flake8 documentation:** https://flake8.pycqa.org/
- **mypy documentation:** https://mypy.readthedocs.io/


## AI Usage

AI tools (ChatGPT) were used as an assistant during the development of this project, mainly for guidance, code review, debugging, and documentation.

AI assistance was used for the following tasks:

- **Code review:** Reviewing the constrained decoding implementation, function selection, parameter generation, and validation logic.
- **Debugging:** Identifying potential errors and edge cases, including unsupported parameter types, missing imports, non-finite floating-point values, and failures during prompt processing.
- **Error handling:** Discussing ways to handle individual prompt failures gracefully and preserve successful results instead of aborting the entire run.
- **Testing:** Suggesting test cases for empty strings, numeric edge cases, invalid function selections, generation failures, and valid JSON output.
- **Documentation:** Assisting with the README structure and descriptions of the project, its usage, design, and resources.

The core project implementation uses Python and the provided `llm_sdk` to perform model inference and constrained decoding. AI suggestions were used as development guidance; proposed changes and code must be reviewed, integrated, and tested by the project author.

The project author remains responsible for the final implementation, correctness, and evaluation results.
