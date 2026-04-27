# Development Guidelines

This document contains critical information about working with this codebase. Follow these guidelines precisely.

## Imports
Always separate imports by type and alphabetically, first standard library imports, then third-party imports, then local imports.

## Functions and Methods
- We like functions of the form:
    `function_name(
        arg1,
        arg2,
        arg3,
    )`

    Note the newlines between the function name and the first argument, and between the last argument and the closing parenthesis.
- Always add a docstring to explain what the function does, its parameters, and what it returns.
- Use type hints for all function parameters and return values.

## Development Philosophy

- **Simplicity**: Write simple, straightforward code
- **Readability**: Make code easy to understand
- **Performance**: Consider performance without sacrificing readability
- **Maintainability**: Write code that's easy to update
- **Testability**: Ensure code is testable
- **Reusability**: Create reusable components and functions
- **Less Code = Less Debt**: Minimize code footprint

## Coding Best Practices

- **Early Returns**: Use to avoid nested conditions
- **Descriptive Names**: Use clear variable/function names (prefix handlers with "handle")
- **Constants Over Functions**: Use constants where possible
- **DRY Code**: Don't repeat yourself
- **Functional Style**: Prefer functional, immutable approaches when not verbose
- **Minimal Changes**: Only modify code related to the task at hand
- **Function Ordering**: Define composing functions before their components
- **TODO Comments**: Mark issues in existing code with "TODO:" prefix
- **Simplicity**: Prioritize simplicity and readability over clever solutions
- **Build Iteratively** Start with minimal functionality and verify it works before adding complexity
- **Run Tests**: Test your code frequently with realistic inputs and validate outputs
- **Build Test Environments**: Create testing environments for components that are difficult to validate directly
- **Functional Code**: Use functional and stateless approaches where they improve clarity
- **Clean logic**: Keep core logic clean and push implementation details to the edges
- **File Organsiation**: Balance file organization with simplicity - use an appropriate number of files for the project scale

## Adding Datasets

- Run `steb new-dataset <name> --type huggingface` or `steb new-dataset <name> --type custom` to scaffold a new dataset.
- Run `steb validate` to check all config.json files are well-formed.
- **Loader location convention**:
  - If a loader is shared by multiple datasets (e.g. PAN, Fisher) -> `steb/loaders/`
  - If a loader is specific to one dataset -> `steb/steb_datasets/<name>/loader.py`
- Every new dataset should have at least one task in its `tasks` field.
- For custom datasets, add the download step to `download_datasets.sh`.
- After the loader and configs are in place, suggest a LaTeX paragraph (4–8 sentences) describing: dataset name + citation, what each record contains, why this dataset was added (style dimension probed), the STEB task(s) it supports, the subset/splits chosen and why, and any caveats (e.g. tokenization).
- Include the LaTeX paragraph in the PR description under a `## LaTeX paragraph` heading so it's easy to lift.

## Pull Requests

- Create a detailed message of what changed. Focus on the high level description of
  the problem it tries to solve, and how it is solved. Don't go into the specifics of the
  code unless it adds clarity.

- Always add `rrivera1849` as reviewer.

- NEVER ever mention a `co-authored-by` or similar aspects. In particular, never
  mention the tool used to create the commit message or PR.
