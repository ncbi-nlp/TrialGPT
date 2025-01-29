# corpus_index.py

import json
import os

import faiss
import numpy as np
import torch
import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_and_format_patient_descriptions(corpus):
    """
    Load and format patient descriptions from a specified corpus.

    This function reads patient descriptions from a JSONL file, formats each description
    by numbering its sentences, and adds a standard closing statement about patient
    compliance and consent.

    Args:
        corpus (str): The name of the corpus (e.g., 'trec_2021', 'trec_2022', 'sigir').

    Returns:
        dict: A dictionary where keys are patient IDs and values are formatted patient descriptions.

    The formatted description includes:
    - Numbered sentences from the original description.
    - An additional sentence about patient consent and compliance.
    """
    query_file = f"dataset/{corpus}/queries.jsonl"
    patients = {}
    with open(query_file, 'r') as f:
        for line in f:
            query = json.loads(line)
            text = query['text']
            sentences = sent_tokenize(text)
            formatted_text = " ".join([f"<{i}.> {sent}" for i, sent in enumerate(sentences)])
            formatted_text += f" <{len(sentences)}.> The patient will provide informed consent, and will comply with the trial protocol without any practical issues."
            patients[query['_id']] = formatted_text
    return patients


def get_bm25_corpus_index(corpus: str, overwrite: bool) -> tuple[BM25Okapi, list[str]]:
    """
    Create or load a BM25 index for a given corpus.

    This function either creates a new BM25 index for the specified corpus or loads
    an existing one, depending on the 'overwrite' flag and whether a cached index exists.

    Args:
        corpus (str): The name of the corpus to index.
        overwrite (bool): If True, always create a new index. If False, use cached index if available.

    Returns:
        tuple: A tuple containing:
            - BM25Okapi: The BM25 index object.
            - list[str]: A list of corpus NCT IDs.

    The function performs the following steps:
    1. Determine the path for the cached corpus index.
    2. If overwrite is True or no cached index exists, create a new index:
       - Tokenize the corpus entries, with weighted emphasis on title and condition.
       - Save the tokenized corpus and NCT IDs to a JSON file.
    3. If a cached index exists and overwrite is False, load the existing index.
    4. Create and return a BM25Okapi object along with the corpus NCT IDs.
    """
    # Define the path for the cached corpus index
    corpus_path = os.path.join(f"results/bm25_corpus_{corpus}.json")

    # Always build the index if overwrite is True or if the cache doesn't exist
    if overwrite or not os.path.exists(corpus_path):
        print("Creating new BM25 index")
        tokenized_corpus = []
        corpus_nctids = []

        # Open and process the corpus file
        with open(f"dataset/{corpus}/corpus.jsonl", "r") as f:
            for line in f.readlines():
                entry = json.loads(line)
                corpus_nctids.append(entry["_id"])

                # Tokenize with weighting: 3 * title, 2 * condition, 1 * text
                tokens = word_tokenize(entry["title"].lower()) * 3
                for disease in entry["metadata"]["diseases_list"]:
                    tokens += word_tokenize(disease.lower()) * 2
                tokens += word_tokenize(entry["text"].lower())

                tokenized_corpus.append(tokens)

        # Prepare data for caching
        corpus_data = {
            "tokenized_corpus": tokenized_corpus,
            "corpus_nctids": corpus_nctids,
        }

        # Cache the processed corpus data
        with open(corpus_path, "w") as f:
            json.dump(corpus_data, f, indent=4)
    else:
        print("Loading existing BM25 index")
        # Load the cached corpus data
        corpus_data = json.load(open(corpus_path))
        tokenized_corpus = corpus_data["tokenized_corpus"]
        corpus_nctids = corpus_data["corpus_nctids"]

    # Create the BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, corpus_nctids

def batch_encode_corpus(corpus: str, batch_size: int = 32) -> tuple[np.ndarray, list[str]]:
    """
    Encode a corpus of documents using the MedCPT-Article-Encoder model in batches.

    This function reads a corpus from a JSONL file, encodes the title and text of each document
    using the MedCPT-Article-Encoder, and returns the embeddings along with the corresponding NCT IDs.

    Args:
        corpus (str): The name of the corpus to encode. This should correspond to a JSONL file
                      located at f"dataset/{corpus}/corpus.jsonl".
        batch_size (int, optional): The number of documents to process in each batch.
                                    Defaults to 32. Adjust based on available GPU memory.

    Returns:
        Tuple[np.ndarray, List[str]]: A tuple containing:
            - np.ndarray: An array of shape (n_documents, embedding_dim) containing the document embeddings.
            - List[str]: A list of NCT IDs corresponding to each embedding.

    Raises:
        FileNotFoundError: If the corpus file cannot be found.
        RuntimeError: If there's an error during the encoding process.

    Note:
        This function requires a CUDA-capable GPU and will move the model and inputs to the GPU.
        Ensure you have sufficient GPU memory for the specified batch size.

    """
    corpus_nctids = []
    titles = []
    texts = []
    embeds = []

    model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

    with open(f"dataset/{corpus}/corpus.jsonl", "r") as f:
        print("Reading corpus")
        for line in tqdm(f, desc="Reading entries"):
            entry = json.loads(line)
            corpus_nctids.append(entry["_id"])
            titles.append(entry["title"])
            texts.append(entry["text"])

    print("Encoding the corpus")
    for i in tqdm(range(0, len(titles), batch_size), desc="Encoding batches"):
        batch_titles = titles[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]

        with torch.no_grad():
            # Tokenize the articles
            encoded = tokenizer(
                list(zip(batch_titles, batch_texts)),
                truncation=True,
                padding=True,
                return_tensors='pt',
                max_length=512,
            ).to("cuda")

            # Generate embeddings
            batch_embeds = model(**encoded).last_hidden_state[:, 0, :]
            embeds.append(batch_embeds.cpu().numpy())

    embeds = np.concatenate(embeds, axis=0)

    return embeds, corpus_nctids

def get_medcpt_corpus_index(corpus: str, overwrite: bool, batch_size: int = 32) -> tuple[faiss.IndexFlatIP, list[str]]:
    """
    Create or load a MedCPT-based corpus index for efficient similarity search.

    This function either generates new embeddings for the corpus using the MedCPT model
    or loads pre-computed embeddings, depending on the 'overwrite' flag and cache existence.
    It then creates a FAISS index for fast similarity search.

    Args:
        corpus (str): The name of the corpus to index. This should correspond to a directory
                      in the 'dataset' folder containing a 'corpus.jsonl' file.
        overwrite (bool): If True, always create new embeddings. If False, use cached embeddings if available.
        batch_size (int, optional): Number of documents to process in each batch when creating new embeddings.
                                    Defaults to 32. Adjust based on available GPU memory.

    Returns:
        tuple[faiss.IndexFlatIP, list[str]]: A tuple containing:
            - faiss.IndexFlatIP: A FAISS index for efficient similarity search.
            - list[str]: A list of corpus NCT IDs corresponding to the indexed documents.

    Raises:
        FileNotFoundError: If the corpus file or cached embeddings cannot be found.
        ValueError: If the loaded embeddings do not match the expected shape or format.

    The function performs the following steps:
    1. Check if cached embeddings exist and whether to overwrite them.
    2. If creating new embeddings:
       - Call batch_encode_corpus to generate embeddings for the corpus documents.
       - Save the embeddings and NCT IDs to disk for future use.
    3. If using cached embeddings, load them from disk.
    4. Create a FAISS index using the embeddings.
    5. Return the FAISS index and the list of NCT IDs.

    Note:
        - This function requires the FAISS library for creating the similarity search index.
        - When creating new embeddings, a CUDA-capable GPU is required.
        - The cached embeddings are stored as '{corpus}_embeds.npy' and '{corpus}_nctids.json' in the 'results' directory.

    """
    corpus_path = f"results/{corpus}_embeds.npy"
    nctids_path = f"results/{corpus}_nctids.json"

    if overwrite or not os.path.exists(corpus_path):
        print("Creating new MedCPT index")
        embeds, corpus_nctids = batch_encode_corpus(corpus, batch_size)

        np.save(corpus_path, embeds)
        with open(nctids_path, "w") as f:
            json.dump(corpus_nctids, f, indent=4)
    else:
        print("Loading existing MedCPT index")
        embeds = np.load(corpus_path)
        with open(nctids_path, "r") as f:
            corpus_nctids = json.load(f)

    index = faiss.IndexFlatIP(embeds.shape[1])
    index.add(embeds)

    return index, corpus_nctids