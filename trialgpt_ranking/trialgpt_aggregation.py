__author__ = "qiao"

"""
TrialGPT-Ranking main functions.
"""

import json
import os

from common.utils import generate_response


def convert_criteria_pred_to_string(
        prediction: dict,
        trial_info: dict,
) -> str:
    """
    Convert TrialGPT prediction to a linear string of criteria.

    This function takes the structured prediction data and trial information,
    and converts it into a more readable, linear string format. It organizes
    the information by inclusion and exclusion criteria, and for each criterion,
    it provides the full text along with the relevance, evident sentences (if any),
    and eligibility predictions.

    Args:
        prediction (dict): A dictionary containing the TrialGPT predictions.
        trial_info (dict): A dictionary containing information about the clinical trial.

    Returns:
        str: A formatted string containing the criteria and their predictions.
    """
    output = ""

    # Process both inclusion and exclusion criteria
    for inc_exc in ["inclusion", "exclusion"]:
        # Create a dictionary to map indices to criteria
        idx2criterion = {}
        # Split the criteria text by double newlines
        criteria = trial_info[inc_exc + "_criteria"].split("\n\n")

        idx = 0
        for criterion in criteria:
            criterion = criterion.strip()

            # Skip headers or invalid criteria
            if "inclusion criteria" in criterion.lower() or "exclusion criteria" in criterion.lower():
                continue
            if len(criterion) < 5:
                continue

            # Add valid criterion to the dictionary
            idx2criterion[str(idx)] = criterion
            idx += 1

        # Process predictions for each criterion
        for idx, info in enumerate(prediction[inc_exc].items()):
            criterion_idx, preds = info

            # Skip criteria not in our dictionary
            if criterion_idx not in idx2criterion:
                continue

            criterion = idx2criterion[criterion_idx]

            # Skip predictions without exactly 3 elements
            if len(preds) != 3:
                continue

            # Build the output string for this criterion
            output += f"{inc_exc} criterion {idx}: {criterion}\n"
            output += f"\tPatient relevance: {preds[0]}\n"
            # Add evident sentences if they exist
            if len(preds[1]) > 0:
                output += f"\tEvident sentences: {preds[1]}\n"
            output += f"\tPatient eligibility: {preds[2]}\n"

    return output


def convert_pred_to_prompt(patient: str, pred: dict, trial_info: dict) -> tuple[str, str]:
    """Generate improved system and user prompts for clinical trial relevance and eligibility scoring."""

    # Construct trial information string
    trial = (
        f"Title: {trial_info['brief_title']}\n"
        f"Target conditions: {', '.join(trial_info['diseases_list'])}\n"
        f"Summary: {trial_info['brief_summary']}"
    )

    # Convert prediction to string
    pred_string = convert_criteria_pred_to_string(pred, trial_info)

    # System prompt
    system_prompt = """You are a clinical trial recruitment specialist. Your task is to assess patient-trial relevance and eligibility based on a patient note, clinical trial description, and criterion-level eligibility predictions.

### Instructions:

1. **Relevance Score (R)**:
   - Provide a detailed explanation of why the patient is relevant (or irrelevant) to the clinical trial.
   - Predict a relevance score (R) between `0` and `100`:
     - `R=0`: The patient is completely irrelevant to the clinical trial.
     - `R=100`: The patient is perfectly relevant to the clinical trial.

2. **Eligibility Score (E)**:
   - Provide a detailed explanation of the patient’s eligibility for the clinical trial.
   - Predict an eligibility score (E) between `-R` and `R`:
     - `E=-R`: The patient is ineligible (meets no inclusion criteria or is excluded by all exclusion criteria).
     - `E=0`: Neutral (no relevant information for any criteria is found).
     - `E=R`: The patient is fully eligible (meets all inclusion criteria and no exclusion criteria).

Prioritize accuracy, use standardized medical terminology in your analysis, and ensure that your scores are within the specified ranges (`0 ≤ R ≤ 100` and `-R ≤ E ≤ R`)."""

    # User prompt
    user_prompt = f"""Analyze the following information:

### Patient Note:
{patient}

### Clinical Trial:
{trial}

### Criterion-level Eligibility Predictions:
{pred_string}

### Output Instructions:
- **Provide ONLY a valid JSON object** with the exact structure below:
```json
{{
  "relevance_explanation": "Your detailed reasoning for the relevance score",
  "relevance_score_R": float_value_between_0_and_100,
  "eligibility_explanation": "Your detailed reasoning for the eligibility score",
  "eligibility_score_E": float_value_between_negative_R_and_positive_R
}}
```

- **Critical Rules:**
  1. **Do NOT include any text outside of the JSON object.**
  2. All reasoning and additional context **MUST** be included in the `relevance_explanation` and `eligibility_explanation` fields.
  3. Ensure that all values are valid:
     - The `relevance_score_R` must be a float between `0` and `100`.
     - The `eligibility_score_E` must be a float in the range of `[-relevance_score_R, +relevance_score_R]`.

### Example Output:
```json
{{
  "relevance_explanation": "The patient has a condition explicitly mentioned in the trial criteria, making this trial highly relevant.",
  "relevance_score_R": 85.0,
  "eligibility_explanation": "The patient meets most inclusion criteria but is disqualified by one exclusion criterion (hypertension).",
  "eligibility_score_E": -20.0
}}
```

- **Additional Notes**:
  - Populate all fields, even if explanations are brief.
  - If there’s no additional information to provide, use an empty string `""` for the explanation fields.
  - **Any text outside the JSON structure will be considered invalid output.**
"""

    return system_prompt, user_prompt


def trialgpt_aggregation(patient: str, trial_results: dict, trial_info: dict, model: str, model_type: str,
                         model_instance: any):
    debug_data = []
    debug_filename = f"results/messages_trialgpt_aggregation.json"
    system_prompt, user_prompt = convert_pred_to_prompt(
        patient,
        trial_results,
        trial_info
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    result = generate_response(model_type, model_instance, messages, model)
    result = result.strip("`").strip("json")

    # Prepare the new debug entry
    debug_entry = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response": result
    }

    # Read existing debug data or create an empty list
    if os.path.exists(debug_filename):
        with open(debug_filename, 'r') as f:
            try:
                debug_data = json.load(f)
            except json.JSONDecodeError:
                debug_data = []
    else:
        debug_data = []

    # Append the new entry
    debug_data.append(debug_entry)

    #TODO: make debug optional, would be slow for production datasets
    # Write the updated debug data back to the file
    with open(debug_filename, 'w') as f:
        json.dump(debug_data, f, indent=4)

    try:
        result = json.loads(result)
    except json.JSONDecodeError:
        print(f"Error parsing JSON: {result}")
        result = {
            "relevance_explanation": "Error parsing JSON",
            "relevance_score_R": 0,
            "eligibility_explanation": "Error parsing JSON",
            "eligibility_score_E": 0
        }

    return result