*This project has been created as part  of the 42 curriculum by ipykhtin*

# Function Calling LLM Project

---

## Description

This project implements a **function-calling pipeline using a small LLM model**. The goal is to generate structured JSON outputs that map user prompts to predefined function calls.

The system:
- Loads function definitions
- Builds a strict system prompt
- Generates model outputs token-by-token
- Validates responses using Pydantic
- Retries generation when invalid JSON is produced

The model is constrained to return only valid JSON in the format:

```json
{"name":"<function_name>", "parameters":{...}}
```

## Instructions

### Setup

Create a virtual enviroment

Install the dependencies
```bash
make install
```

2. Run the project
```bash
make run
```
* Optional flags -v or --visualize

## Algoritm explaination

### Constrained Decoding Approach
*	1.	A strict system prompt is constructed containing all available functions.
*	2.	The model generates output token-by-token.
*	3.	At each step: Logits are computed and the most probable token is selected using np.argmax
*	4.	Generation stops when a valid JSON object is detected.
*	5.	If invalid - The system modifies the prompt and retries up to MAX_RETRIES

### Key Constraint
Output must strictly match:
```json
{"name":"<function_name>", "parameters":{...}}
```

##  Design decisions
* Greedy decoding (argmax)
* Ensures deterministic and stable JSON generation.
* Strict prompt engineering
* Prevents the model from adding explanations or extra text.
* Improves robustness when model outputs malformed JSON.
* Validation layer (Pydantic)
* Handles different outputs like:

## Performance analysis
* Programm consistently return 100% valid json
* Runtime harshly depends on the input prompts but for the ones given it is ~4 minutes
* Robust against malformed outputs

## Challenges
1. Quote handling ("" vs '')
    * One of the difficulties that I faced is the difference with "" and ''. I spent the whole day trying to find the mistake in:
```json
"prompt": "Replace all vowels in 'Programming is fun' with asterisks"
```
The issue was caused by inconsistent quote usage.

2. Memory optimization
* Due to insufficient usage of memory and weak prompt engineering. My OS crashed several times because the model was stuck generating tensors endlessly

Solution:
* Reduced prompt size
* Made instructions more explicit and minimal

3. Invalid JSON output
Model sometimes returned:
* "Result is ..."
* "Here is your output"

or other garbage data

Solution:
* Implemented JSON extraction
* Added validation layer
* Introduced retry mechanism

4. Prompt experimentation
I have also experimented with the most efficient system_prompt that can bring me to the most consistent output

## Testing strategy
1. Used function given data.zip
2. Tried my personal edge cases

# Resources
1. 42 Abu Dhabi subject
2. Gemini:
* A brief research on Machine learning concept and all the corresponding topics,
* Help with debugging
* explaination of a complicated structures like Tensors

Most of the research was made on a llm_sdk project rather on my project itself
