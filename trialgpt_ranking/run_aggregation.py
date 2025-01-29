__author__ = "qiao"

"""
Using GPT or Llama to aggregate the scores by itself.
"""

import argparse
import json
import os
import sys

from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from trialgpt_aggregation import trialgpt_aggregation

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import load_corpus_details, setup_model, generate_response


def parse_arguments():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description="Run TrialGPT aggregation for specified corpus and model.")
    parser.add_argument("corpus", help="Corpus name")
    parser.add_argument("model", help="Model to use for aggregation")
    parser.add_argument("matching_results_path", help="Path to the matching results file")
    parser.add_argument("overwrite", help="Overwrite existing results (true/false)")
    parser.add_argument("-g", "--num_gpus", help="The number of GPUs to use for model distribution")
    parser.add_argument("-d", "--checkpoint_dir", help="Checkpoint directory for Llama models")
    parser.add_argument("-q", "--quantize", action="store_true", help="Use 8-bit quantization for Llama models")
    return parser.parse_args()

def load_data(retrieved_trials_path, corpus_path):
    """
    Load necessary data from files.

    Args:
        retrieved_trials_path (str): Path to the retrieved trials JSON file.
        corpus_path (str): Path to the corpus JSONL file.

    Returns:
        tuple: Contains retrieved trials data and trial info dictionary.
    """
    with open(retrieved_trials_path, 'r') as f:
        retrieved_trials = json.load(f)

    trial2info = load_corpus_details(corpus_path)

    return retrieved_trials, trial2info


def main(args):
    """
    Main function to run the TrialGPT aggregation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    retrieved_trials, trial2info = load_data(
        f"results/retrieved_trials.json",
        f"dataset/{args.corpus}/corpus.jsonl"
    )

    output_path = f"results/aggregation_results_{args.corpus}_{args.model}.json"
    failed_output_path = f"results/failed_aggregation_results_{args.corpus}_{args.model}.json"

    if args.overwrite.lower() == 'true' or not os.path.exists(output_path):
        print(f"Creating new aggregation results for {args.corpus} with {args.model}")
        output = {}
        failed_outputs = {}
    else:
        print(f"Loading existing aggregation results for {args.corpus} with {args.model}")
        with open(output_path, 'r') as f:
            output = json.load(f)
        if os.path.exists(failed_output_path):
            with open(failed_output_path, 'r') as f:
                failed_outputs = json.load(f)
        else:
            failed_outputs = {}

    results = json.load(open(args.matching_results_path))

    # Set up the model
    model_type, model_instance = setup_model(args.model, args.num_gpus, args.checkpoint_dir, args.quantize)

    for patient_entry in tqdm(retrieved_trials, desc="Processing patients"):
        patient_id = patient_entry["patient_id"]
        patient = patient_entry["patient"]
        # already done in the data set from load_and_format_patient_descriptions(corpus)
        # sents = sent_tokenize(patient)
        # sents.append(
        #     "The patient will provide informed consent, and will comply with the trial protocol without any practical issues.")
        # sents = [f"<{idx}.> {sent}" for idx, sent in enumerate(sents)]
        # patient = "\n".join(sents)

        if patient_id not in output:
            output[patient_id] = {}
        if patient_id not in failed_outputs:
            failed_outputs[patient_id] = {}

        # TODO: remove the 2 1 0 loop its irrelevant at this stage, but must be removed throughout at the same time.
        # Middle progress bar for labels
        for label in tqdm(["2", "1", "0"], desc=f"Labels for patient {patient_id}", leave=False):
            if label not in patient_entry:
                continue

            # Inner progress bar for trials
            for trial in tqdm(patient_entry[label], desc=f"Trials for label {label}", leave=False):
                trial_id = trial["NCTID"]

                # Skip if already processed and not overwriting
                if args.overwrite.lower() != 'true' and trial_id in output[patient_id]:
                    continue

                trial_results = results.get(patient_id, {}).get(label, {}).get(trial_id)

                if not isinstance(trial_results, dict):
                    failed_outputs[patient_id][trial_id] = "matching result error"
                    continue

                trial_info = trial2info.get(trial_id, {})

                if not trial_info:
                    print(f"Warning: No trial info found for trial {trial_id}")
                    failed_outputs[patient_id][trial_id] = "no trial info"
                    continue

                try:
                    result = trialgpt_aggregation(patient, trial_results, trial_info, args.model, model_type, model_instance)
                    output[patient_id][trial_id] = result
                except Exception as e:
                    print(f"Error processing trial {trial_id} for patient {patient_id}: {e}")
                    failed_outputs[patient_id][trial_id] = str(e)
                    continue

        # Save after each patient to ensure progress is not lost
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)
        with open(failed_output_path, "w") as f:
            json.dump(failed_outputs, f, indent=4)

    print(f"Aggregation results saved to {output_path}")
    print(f"Failed aggregation results saved to {failed_output_path}")

    # Print summary
    total_processed = sum(len(patient_data) for patient_data in output.values())
    total_failed = sum(len(patient_data) for patient_data in failed_outputs.values())
    print(f"Total trials processed: {total_processed + total_failed}")
    print(f"Successful trials: {total_processed}")
    print(f"Failed trials: {total_failed}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)