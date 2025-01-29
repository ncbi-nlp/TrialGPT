__author__ = "qiao"

"""
Conduct the first stage retrieval by the hybrid retriever 
"""
#nltk.download('punkt')

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from nltk.tokenize import word_tokenize
from transformers import AutoModel, AutoTokenizer

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import load_corpus_details
from corpus_index import get_bm25_corpus_index, get_medcpt_corpus_index, load_and_format_patient_descriptions


def parse_arguments():
    """
    Parse command-line arguments for the keyword generation script.

    This function sets up the argument parser and defines the required and optional
    arguments for the script.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Hybrid Fusion Retrieval for Clinical Trials")
    parser.add_argument("corpus", help="Path to the corpus")
    parser.add_argument("q_type", help="Model type (e.g., gpt-4o-mini)")
    parser.add_argument("k", type=int, help="Parameter for score calculation")
    parser.add_argument("bm25_wt", type=float, help="Weight for BM25 scores")
    parser.add_argument("medcpt_wt", type=float, help="Weight for MedCPT scores")
    parser.add_argument("overwrite", type=str, help="Overwrite existing index (true/false)")
    parser.add_argument("top_k", type=int, help="Top K documents to consider (default: 1000)")
    parser.add_argument("batch_size", type=int, help="Batch size for processing (default: 32)")
    parser.add_argument("eligibility_threshold", type=float, help="Score threshold for eligibility (0-1)")
    parser.add_argument("exclusion_threshold", type=float, help="Score threshold for exclusion (0-1)")
    return parser.parse_args()


def load_queries(corpus, q_type):
    """
    Load queries from the specified file.

    Args:
        corpus (str): The name of the corpus.
        q_type (str): The type of queries.

    Returns:
        dict: A dictionary containing the loaded queries.
    """
    return json.load(open(f"results/retrieval_keywords_{q_type}_{corpus}.json"))


def load_medcpt_model():
    """Load and return the MedCPT query encoder model and tokenizer."""
    model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    return model, tokenizer


def perform_bm25_search(bm25, bm25_nctids, conditions, N):
    """
    Perform BM25 search for each condition.

    Args:
        bm25: The BM25 index.
        bm25_nctids (list): List of NCT IDs corresponding to the BM25 index.
        conditions (list): List of condition strings to search for.
        N (int): Number of top results to return for each condition.

    Returns:
        list: A list of lists, where each inner list contains the top N NCT IDs for a condition.
    """
    return [
        bm25.get_top_n(word_tokenize(condition.lower()), bm25_nctids, n=N)
        for condition in conditions
    ]


def perform_medcpt_search(model, tokenizer, medcpt, medcpt_nctids, conditions, N):
    """
    Perform MedCPT search for each condition.

    Args:
        model: The MedCPT model.
        tokenizer: The MedCPT tokenizer.
        medcpt: The MedCPT index.
        medcpt_nctids (list): List of NCT IDs corresponding to the MedCPT index.
        conditions (list): List of condition strings to search for.
        N (int): Number of top results to return for each condition.

    Returns:
        list: A list of lists, where each inner list contains the top N NCT IDs for a condition.
    """
    with torch.no_grad():
        encoded = tokenizer(
            conditions,
            truncation=True,
            padding=True,
            return_tensors='pt',
            max_length=512,
        ).to("cuda")
        # encode the queries (use the [CLS] last hidden states as the representations)
        embeds = model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()
        # search the Faiss index
        scores, inds = medcpt.search(embeds, k=N)

    return [[medcpt_nctids[ind] for ind in ind_list] for ind_list in inds]


def calculate_scores(bm25_results, medcpt_results, k, bm25_wt, medcpt_wt):
    """
    Calculate combined scores for BM25 and MedCPT results.

    Args:
        bm25_results (list): Results from BM25 search.
        medcpt_results (list): Results from MedCPT search.
        k (int): Parameter for score calculation.
        bm25_wt (float): Weight for BM25 scores.
        medcpt_wt (float): Weight for MedCPT scores.

    Returns:
        tuple: A tuple containing two dictionaries:
            - nctid2score: Maps NCT IDs to their total scores.
            - nctid2details: Maps NCT IDs to dictionaries containing separate BM25 and MedCPT scores.
    """
    nctid2score = {}
    nctid2details = {}
    for condition_idx, (bm25_top_nctids, medcpt_top_nctids) in enumerate(zip(bm25_results, medcpt_results)):
        condition_weight = 1 / (condition_idx + 1)

        if bm25_wt > 0:
            for rank, nctid in enumerate(bm25_top_nctids):
                bm25_score = (1 / (rank + k)) * condition_weight * bm25_wt
                if nctid not in nctid2score:
                    nctid2score[nctid] = 0
                    nctid2details[nctid] = {"bm25_score": 0, "medcpt_score": 0}
                nctid2score[nctid] += bm25_score
                nctid2details[nctid]["bm25_score"] += bm25_score

        if medcpt_wt > 0:
            for rank, nctid in enumerate(medcpt_top_nctids):
                medcpt_score = (1 / (rank + k)) * condition_weight * medcpt_wt
                if nctid not in nctid2score:
                    nctid2score[nctid] = 0
                    nctid2details[nctid] = {"bm25_score": 0, "medcpt_score": 0}
                nctid2score[nctid] += medcpt_score
                nctid2details[nctid]["medcpt_score"] += medcpt_score

    return nctid2score, nctid2details


def assign_relevance_labels(retrieved_trials, eligibility_threshold, exclusion_threshold):
    # TODO: remove this as it's irrelevant at this stage, but must be removed throughout at the same time.
    """
    Assign relevance labels to retrieved trials based on the TREC Clinical Trials track criteria.

    Args:
        retrieved_trials (dict): Dictionary of retrieved trials with their scores.
        eligibility_threshold (float): Score threshold for eligibility (0-1).
        exclusion_threshold (float): Score threshold for exclusion (0-1).

    Returns:
        defaultdict: A nested defaultdict where the outer key is the query ID,
                     the inner key is the relevance label (0, 1, or 2),
                     and the value is a list of trials with that label.

    Note:
        The function assigns labels as follows:
        - 2: Eligible - The patient is eligible to enroll in the trial.
        - 1: Excluded - The patient has the condition that the trial is targeting,
                        but the exclusion criteria make the patient ineligible.
        - 0: Not Relevant - The patient is not relevant for the trial in any way.
    """
    # labeled_trials = defaultdict(lambda: defaultdict(list))
    # for qid, trials in retrieved_trials.items():
    #     for trial in trials:
    #         score = trial['total_score']
    #         if score >= eligibility_threshold:
    #             labeled_trials[qid]['2'].append(trial)
    #         elif score >= exclusion_threshold:
    #             labeled_trials[qid]['1'].append(trial)
    #         else:
    #             labeled_trials[qid]['0'].append(trial)
    #
    # return labeled_trials

    labeled_trials = defaultdict(lambda: defaultdict(list))
    for qid, trials in retrieved_trials.items():
        labeled_trials[qid]['0'] = trials

    return labeled_trials


def main(args):
    """
    Main function to perform hybrid fusion retrieval and generate retrieved trials.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    This function:
    1. Loads necessary data and models
    2. Performs hybrid fusion retrieval using BM25 and MedCPT
    3. Calculates combined scores for retrieved trials
    4. Assigns relevance labels to the retrieved trials
    5. Saves the final output as a JSON file
    """
    corpus_details = load_corpus_details(f"dataset/{args.corpus}/corpus.jsonl")
    bm25, bm25_nctids = get_bm25_corpus_index(args.corpus, args.overwrite)
    medcpt, medcpt_nctids = get_medcpt_corpus_index(args.corpus, args.overwrite, args.batch_size)
    patient_descriptions = load_and_format_patient_descriptions(args.corpus)
    id2queries = load_queries(args.corpus, args.q_type)
    model, tokenizer = load_medcpt_model()

    # Perform hybrid fusion retrieval
    retrieved_trials = {}
    for qid, patient_desc in patient_descriptions.items():
        conditions = id2queries[qid]["conditions"]
        if not conditions:
            retrieved_trials[qid] = []
            continue

        bm25_results = perform_bm25_search(bm25, bm25_nctids, conditions, args.top_k)
        medcpt_results = perform_medcpt_search(model, tokenizer, medcpt, medcpt_nctids, conditions, args.top_k)
        nctid2score, nctid2details = calculate_scores(bm25_results, medcpt_results, args.k, args.bm25_wt,
                                                      args.medcpt_wt)
        top_nctids = sorted(nctid2score.items(), key=lambda x: -x[1])[:args.top_k]
        retrieved_trials[qid] = [
            {
                "nct_id": nctid,
                **corpus_details[nctid],
                "total_score": score,
                "bm25_score": nctid2details[nctid]["bm25_score"],
                "medcpt_score": nctid2details[nctid]["medcpt_score"]
            }
            for nctid, score in top_nctids
        ]

    # Generate retrieved trials with relevance labels
    labeled_trials = assign_relevance_labels(retrieved_trials, args.eligibility_threshold, args.exclusion_threshold)

    final_output = []
    # TODO: remove the 2 1 0 loop its irrelevant at this stage, but must be removed throughout at the same time.
    for qid, trials_by_label in labeled_trials.items():
        patient_entry = {
            "patient_id": qid,
            "patient": patient_descriptions[qid],
            "0": trials_by_label['0'],
            "1": trials_by_label['1'],
            "2": trials_by_label['2']
        }
        final_output.append(patient_entry)

    # Save retrieved trials with relevance labels
    output_path = f"results/retrieved_trials.json"
    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=4)

    print(f"Retrieved trials saved to {output_path}")

    # Calculate and print the requested statistics
    total_trials = sum(len(patient['0'] + patient['1'] + patient['2']) for patient in final_output)
    eligible_count = sum(len(patient['2']) for patient in final_output)
    excluded_count = sum(len(patient['1']) for patient in final_output)
    not_relevant_count = sum(len(patient['0']) for patient in final_output)

    print(f"Total trials processed: {total_trials}")
    print(f"Eligible trials: {eligible_count} ({eligible_count / total_trials:.2%})")
    print(f"Excluded trials: {excluded_count} ({excluded_count / total_trials:.2%})")
    print(f"Not relevant trials: {not_relevant_count} ({not_relevant_count / total_trials:.2%})")

    # Calculate total_score distribution
    all_scores = [trial['total_score'] for patient in final_output for trials in patient.values() if
                  isinstance(trials, list) for trial in trials]
    all_scores.sort()

    percentiles = [0, 10, 25, 33, 50, 66, 70, 80, 90, 100]
    percentile_values = [np.percentile(all_scores, p) for p in percentiles]

    print("\nTotal Score Distribution:")
    print(f"Min: {min(all_scores):.4f}")
    print(f"Max: {max(all_scores):.4f}")
    print(f"Average: {np.mean(all_scores):.4f}")

    # Calculate mode manually
    from collections import Counter
    mode_value = Counter(all_scores).most_common(1)[0][0]
    print(f"Mode: {mode_value:.4f}")

    print("Percentiles:")
    for p, v in zip(percentiles, percentile_values):
        print(f"{p}th: {v:.4f}")

if __name__ == "__main__":
    args = parse_arguments()

    # Convert overwrite string to boolean
    args.overwrite = args.overwrite.lower() == 'true'

    main(args)
