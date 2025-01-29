# TrialGPT/trialgpt_matching/run_matching.py

__author__ = "qiao"

"""
Running the TrialGPT matching for three cohorts (sigir, TREC 2021, TREC 2022).
"""

import argparse
import json
import os
import sys

from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from TrialGPT import trialgpt_match

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import setup_model

def parse_arguments():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description="Run TrialGPT matching for specified corpus and model.")
    parser.add_argument("corpus", help="Corpus name")
    parser.add_argument("model", help="Model to use for matching")
    parser.add_argument("overwrite", help="Overwrite existing results (true/false)")
    parser.add_argument("-g", "--num_gpus", help="The number of GPUs to use for model distribution")
    parser.add_argument("-d", "--checkpoint_dir", help="Checkpoint directory for Llama models")
    parser.add_argument("-q", "--quantize", action="store_true", help="Use 8-bit quantization for Llama models")
    return parser.parse_args()


def main(args):
    dataset = json.load(open(f"results/retrieved_trials.json"))
    output_path = f"results/matching_results_{args.corpus}_{args.model}.json"
    failed_output_path = f"results/failed_matching_results_{args.corpus}_{args.model}.json"

    # Set up the model once before the main processing loop
    model_type, model_instance = setup_model(args.model, args.num_gpus, args.checkpoint_dir, args.quantize)

    # Dict{Str(patient_id): Dict{Str(label): Dict{Str(trial_id): Str(output)}}}
    if args.overwrite.lower() == 'true' or not os.path.exists(output_path):
        print(f"Creating new matching results for {args.corpus} with {args.model}")
        output = {}
    else:
        print(f"Loading existing matching results for {args.corpus} with {args.model}")
        output = json.load(open(output_path))

    failed_outputs = {}

    # Outer progress bar for patients
    for instance in tqdm(dataset, desc="Processing patients", unit="patient"):
        patient_id = instance["patient_id"]
        patient = instance["patient"]
        # already done in the data set from load_and_format_patient_descriptions(corpus)
        # sents = sent_tokenize(patient)
        # sents.append(
        #     "The patient will provide informed consent, and will comply with the trial protocol without any practical issues.")
        # sents = [f"<{idx}.> {sent}" for idx, sent in enumerate(sents)]
        # patient = "\n".join(sents)


        if patient_id not in output:
            output[patient_id] = {"0": {}, "1": {}, "2": {}}

        # Middle progress bar for labels
        #TODO: remove the 2 1 0 loop its irrelevant at this stage, but must be removed throughout at the same time.
        for label in tqdm(["2", "1", "0"], desc=f"Labels for patient {patient_id}", leave=False):
            if label not in instance: continue

            # Inner progress bar for trials
            for trial in tqdm(instance[label], desc=f"Trials for label {label}", leave=False):
                trial_id = trial["NCTID"]

                if args.overwrite.lower() != 'true' and trial_id in output[patient_id][label]:
                    continue

                try:
                    results = trialgpt_match(
                        trial,
                        patient,
                        args.model,
                        model_type,
                        model_instance
                    )
                    output[patient_id][label][trial_id] = results

                    # Save after each trial to ensure progress is not lost
                    with open(output_path, "w") as f:
                        json.dump(output, f, indent=4)

                except Exception as e:
                    error_message = f"Error processing trial {trial_id} for patient {patient_id}: {str(e)}"
                    print(error_message)
                    if patient_id not in failed_outputs:
                        failed_outputs[patient_id] = {}
                    if label not in failed_outputs[patient_id]:
                        failed_outputs[patient_id][label] = {}
                    failed_outputs[patient_id][label][trial_id] = {
                        "error": str(e),
                        "patient": patient,
                        "trial": trial
                    }

                    # Save failed outputs after each error
                    with open(failed_output_path, "w") as f:
                        json.dump(failed_outputs, f, indent=4)

                    continue

    print(f"Matching results saved to {output_path}")
    print(f"Failed matching results saved to {failed_output_path}")

    # Print summary
    total_processed = sum(
        sum(len(label_data) for label_data in patient_data.values()) for patient_data in output.values())
    total_failed = sum(
        sum(len(label_data) for label_data in patient_data.values()) for patient_data in failed_outputs.values())
    print(f"Total trials processed: {total_processed + total_failed}")
    print(f"Successful trials: {total_processed}")
    print(f"Failed trials: {total_failed}")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)