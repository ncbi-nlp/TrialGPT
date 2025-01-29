# trialgpt_retrieval/__init__.py

import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import load_corpus_details

from .corpus_index import (
    load_and_format_patient_descriptions,
    get_bm25_corpus_index,
    batch_encode_corpus,
    get_medcpt_corpus_index
)

from .hybrid_fusion_retrieval import (
    parse_arguments,
    load_queries,
    load_medcpt_model,
    perform_bm25_search,
    perform_medcpt_search,
    calculate_scores,
    assign_relevance_labels
)

from .keyword_generation import (
    parse_arguments_kg,
    get_keyword_generation_messages,
)

__all__ = [
    'load_corpus_details',
    'load_and_format_patient_descriptions',
    'get_bm25_corpus_index',
    'batch_encode_corpus',
    'get_medcpt_corpus_index',
    'parse_arguments',
    'load_queries',
    'load_medcpt_model',
    'perform_bm25_search',
    'perform_medcpt_search',
    'calculate_scores',
    'assign_relevance_labels',
    'parse_arguments_kg',
    'get_keyword_generation_messages'
]