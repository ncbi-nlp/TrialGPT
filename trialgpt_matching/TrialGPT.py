__author__ = "qiao"

"""
TrialGPT-Matching main functions.
"""

import json
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import generate_response


def parse_criteria(criteria: str) -> str:
    """
    Parse and format a specific set of clinical trial criteria (either inclusion or exclusion).

    This function takes a string containing either inclusion or exclusion criteria
    for a clinical trial and formats it into a numbered list. It removes any header
    mentioning "inclusion criteria" or "exclusion criteria" and skips empty or very short lines.

    Args:
        criteria (str): A string containing either inclusion or exclusion criteria.

    Returns:
        str: A formatted string with numbered criteria, excluding headers and empty lines.

    Example:
        Input:
        '''
        Inclusion Criteria:

        - Age 18 or older
        - Diagnosed with Type 2 Diabetes
        '''

        Output:
        '''
        0. Age 18 or older
        1. Diagnosed with Type 2 Diabetes
        '''
    """
    output = ""

    # Split the criteria into separate lines
    criteria_lines = criteria.split("\n\n")

    idx = 0
    for line in criteria_lines:
        # Remove leading/trailing whitespace from each line
        line = line.strip()

        # Skip the header line
        if "inclusion criteria" in line.lower() or "exclusion criteria" in line.lower():
            continue

        # Skip lines that are too short (likely empty or irrelevant)
        if len(line) < 5:
            continue

        # Add the formatted criterion to the output
        output += f"{idx}. {line}\n"
        idx += 1

    return output


def print_trial(
    trial_info: dict,
    inc_exc: str,
) -> str:
    """Given a dict of trial information, returns a string of trial."""

    trial = f"Title: {trial_info['brief_title']}\n"
    trial += f"Target diseases: {', '.join(trial_info['diseases_list'])}\n"
    trial += f"Interventions: {', '.join(trial_info['drugs_list'])}\n"
    trial += f"Summary: {trial_info['brief_summary']}\n"

    if inc_exc == "inclusion":
        trial += "Inclusion criteria:\n %s\n" % parse_criteria(trial_info['inclusion_criteria'])
    elif inc_exc == "exclusion":
        trial += "Exclusion criteria:\n %s\n" % parse_criteria(trial_info['exclusion_criteria'])

    return trial


def get_matching_prompt(
    trial_info: dict,
    inc_exc: str,
    patient: str,
) -> tuple[str, str]:
    """Generate the system and user prompts for clinical trial matching."""

    # System prompt: Define the assistant's role and task
    system_prompt = f"""
You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient note
and the {inc_exc} criteria of a clinical trial to determine the patient's eligibility at the criterion level.
Prioritize accuracy and relevance in your analysis.
"""

    # Add inclusion/exclusion-specific context
    if inc_exc == "inclusion":
        system_prompt += """
Inclusion criteria are the factors that allow someone to participate in a clinical study.
They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.
"""
    elif inc_exc == "exclusion":
        system_prompt += """
Exclusion criteria are the factors that disqualify someone from participating in a clinical study.
They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.
"""

    # Add task-specific instructions
    system_prompt += f"""
For each {inc_exc} criterion, you should perform the following steps:
1. **Reasoning**: Determine if the criterion is applicable to the patient. If applicable, check if the patient note contains direct evidence.
   If no direct evidence is found, infer from existing evidence and decide if the criterion is likely true or false.
   If the criterion is not mentioned and is medically important, assume it is not true for the patient.
2. **Relevant Sentences**: Identify the sentence IDs (example:<2.> or <13.>)in the patient note that are relevant to the criterion. If no relevant information is found, return an empty list.
3. **Eligibility Label**: Classify the patient's eligibility for the {inc_exc} criterion using one of the following labels:
"""

    # Add inclusion/exclusion-specific labels
    if inc_exc == "inclusion":
        system_prompt += """
   - "not applicable": The criterion is not relevant to the patient.
   - "not enough information": The patient note lacks sufficient information to make a determination.
   - "included": The patient meets the inclusion criterion.
   - "not included": The patient does not meet the inclusion criterion.
"""
    elif inc_exc == "exclusion":
        system_prompt += """
   - "not applicable": The criterion is not relevant to the patient.
   - "not enough information": The patient note lacks sufficient information to make a determination.
   - "excluded": The patient meets the exclusion criterion and should be excluded from the trial.
   - "not excluded": The patient does not meet the exclusion criterion.
"""

    # Add JSON output instructions with an emphasized "notes" field
    system_prompt += """
### Output Instructions:
- **Provide ONLY a valid JSON object** with the following structure:
```json
{
  "criterion_number": [
    "brief_reasoning",
    [sentence_id_1, sentence_id_2, ...],
    "eligibility_label"
  ],
  "criterion_number": [
    ...
  ]
}
```
- **Do NOT include any text outside of the JSON object.** This means no explanations, headers, or footers outside the JSON.

### Example Output:
```json
{
  "0": [
    "The patient is...",
    [2],
    "included"
  ],
  "1": [
    "The patient has...",
    [3, 5],
    "excluded"
  ]
}
```
"""

    # User prompt: Provide the patient note and trial criteria
    user_prompt = f"""
Here is the patient note, with each sentence labeled by a sentence ID:
{patient}

Here is the clinical trial {inc_exc} criteria:
{print_trial(trial_info, inc_exc)}

Please analyze the information and provide the JSON output as described above.
"""

    return system_prompt, user_prompt

def trialgpt_match(trial: dict[str, any], patient: str, model: str, model_type: str, model_instance: any) -> dict[
    str, any]:
    """
    Match a patient to a clinical trial using GPT-based analysis.

    This function evaluates a patient's eligibility for a clinical trial by analyzing
    both inclusion and exclusion criteria against the patient's information. It uses
    a specified language model to perform the analysis and generates detailed results.

    Args:
        trial (dict[str, any]): A dictionary containing clinical trial information.
        patient (str): A string containing the patient's medical information.
        model (str): The specific GPT model name (for OpenAI models).
        model_type (str): Either 'gpt' or 'llama'.
        model_instance: Either an OpenAI client or a Llama pipeline.


    Returns:
        dict[str, any]: A dictionary containing the matching results for both
                        inclusion and exclusion criteria.

    Side Effects:
        - Writes debug information to a JSON file named "results/messages_TrialGPT.json".
        - Creates the file if it doesn't exist, or appends to it if it does.

    Note:
        This function relies on external utility functions like `get_matching_prompt`
        and `generate_response`, which should be imported or defined elsewhere in the script.
    """
    # Initialize the results dictionary
    results = {}

    # Define the debug file path
    debug_filename = f"results/messages_TrialGPT.json"

    # Read existing debug data or create an empty list
    if os.path.exists(debug_filename):
        with open(debug_filename, 'r') as f:
            try:
                debug_data = json.load(f)
            except json.JSONDecodeError:
                debug_data = []
    else:
        debug_data = []

    # Iterate over inclusion and exclusion criteria
    for inc_exc in ["inclusion", "exclusion"]:
        # Generate prompts for the current criteria type
        system_prompt, user_prompt = get_matching_prompt(trial, inc_exc, patient)

        # Prepare messages for the model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Generate a response using the specified model
        message = generate_response(model_type, model_instance, messages, model)

        # Clean up the response by removing potential markdown code block syntax
        message = message.strip("`").strip("json")

        # Prepare the debug entry for this iteration
        debug_entry = {
            "type": inc_exc,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": message
        }
        debug_data.append(debug_entry)

        # Parse the model's response and store it in the results
        try:
            results[inc_exc] = json.loads(message)
        except json.JSONDecodeError:
            # If JSON parsing fails, store the raw message
            results[inc_exc] = message

    #TODO: make debug optional, would be slow for production datasets
    # Write the updated debug data back to the file
    with open(debug_filename, 'w') as f:
        json.dump(debug_data, f, indent=4)

    return results