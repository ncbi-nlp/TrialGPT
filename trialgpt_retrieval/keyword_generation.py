#!/usr/bin/env python3

"""
Generate search keywords for patient descriptions using specified model and corpus.
"""

import argparse
import json
import os
import sys

from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import setup_model, generate_response


def parse_arguments_kg():
    """
    Parse command-line arguments for the keyword generation script.

    This function sets up the argument parser and defines the required and optional
    arguments for the script.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate search keywords for patient descriptions.")

    # Required arguments
    parser.add_argument("-c", "--corpus", required=True, help="The corpus to process: trec_2021, trec_2022, or sigir")
    parser.add_argument("-m", "--model", required=True, help="The model to use for generating keywords")
    parser.add_argument("-g", "--num_gpus", help="The number of GPUs to use for model distribution")
    # Optional arguments
    parser.add_argument("-d", "--checkpoint_dir", help="Checkpoint directory for Llama models")
    parser.add_argument("-q", "--quantize", action="store_true", help="Use 8-bit quantization for Llama models")

    return parser.parse_args()


def get_keyword_generation_messages(note):
    """
    Prepare the messages for keyword generation based on a patient note.

    Args:
        note (str): The patient description.

    Returns:
        list: A list of message dictionaries for the AI model.
    """
    system = """You are a medical research assistant specializing in clinical trial matching. Your task is to analyze patient descriptions to identify key medical conditions and assist in finding suitable clinical trials. Prioritize accuracy and relevance in your analysis."""

    prompt = f"""Please analyze the following patient description for clinical trial matching:

    ## {note}

    ### Instructions:
    1. Summarize the patient's main medical issues in 3-5 sentences.
    2. List up to 20 key medical conditions, ranked by relevance for clinical trial matching.
    3. Use standardized medical terminology (e.g., "Type 2 Diabetes" instead of "high blood sugar").
    4. Include conditions only if explicitly mentioned or strongly implied in the description.

    ### Output a JSON object in this format:
    **Provide ONLY a valid JSON object** with the following structure:
    {{
      "summary": "Brief patient summary",
      "conditions": ["Condition 1", "Condition 2", ...]
    }}

    ### Important Notes:
    - If you are unsure about a condition, include it only if it is explicitly mentioned or strongly implied in the description.
    - **Do NOT include any text outside of the JSON object.** This means no notes, explanations, headers, or footers outside the JSON.

    Now, please process the patient description and respond with the JSON object.
    """

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]


def main(args):
    """
    Generate search keywords for patient descriptions using specified model and corpus.

    This function processes patient descriptions from a given corpus using either GPT or Llama models
    to generate relevant medical keywords. It saves the results to a JSON file.
    """
    outputs = {}
    failed_outputs = {}

    model_type, model_instance = setup_model(args.model, args.num_gpus, args.checkpoint_dir, args.quantize)

    # Count total lines in the input file for progress tracking
    with open(f"dataset/{args.corpus}/queries.jsonl", "r") as f:
        total_lines = sum(1 for _ in f)

    # Process each query in the input file
    with open(f"dataset/{args.corpus}/queries.jsonl", "r") as f:
        for line in tqdm(f, total=total_lines, desc=f"Processing {args.corpus} queries"):
            try:
                entry = json.loads(line)
                messages = get_keyword_generation_messages(entry["text"])
                output = generate_response(model_type, model_instance, messages, args.model)

                try:
                    outputs[entry["_id"]] = json.loads(output)
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON for entry {entry['_id']}. Output: {output}")
                    failed_outputs[entry["_id"]] = {
                        "error": "Failed to parse JSON",
                        "raw_output": output
                    }
            except Exception as e:
                print(f"Error processing entry {entry['_id']}: {str(e)}")
                failed_outputs[entry["_id"]] = {
                    "error": str(e),
                    "raw_entry": line
                }

    # Save successful outputs
    output_file = f"results/retrieval_keywords_{args.model}_{args.corpus}.json"
    with open(output_file, "w") as f:
        json.dump(outputs, f, indent=4)
    print(f"Results saved to {output_file}")

    # Save failed outputs
    failed_output_file = f"results/failed_retrieval_keywords_{args.model}_{args.corpus}.json"
    with open(failed_output_file, "w") as f:
        json.dump(failed_outputs, f, indent=4)
    print(f"Failed results saved to {failed_output_file}")

    # Print summary
    print(f"Total entries processed: {len(outputs) + len(failed_outputs)}")
    print(f"Successful entries: {len(outputs)}")
    print(f"Failed entries: {len(failed_outputs)}")


if __name__ == "__main__":
    args = parse_arguments_kg()
    main(args)
