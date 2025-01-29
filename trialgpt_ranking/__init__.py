# trialgpt_ranking/__init__.py
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .rank_results import (
    parse_arguments as rank_parse_arguments,
    get_matching_score,
    get_agg_score,
    main as rank_main
)

from .run_aggregation import (
    parse_arguments as agg_parse_arguments,
    load_data,
    main as agg_main
)

from .trialgpt_aggregation import (
    convert_criteria_pred_to_string,
    convert_pred_to_prompt,
    trialgpt_aggregation
)

__all__ = [
    'rank_parse_arguments',
    'get_matching_score',
    'get_agg_score',
    'rank_main',
    'agg_parse_arguments',
    'load_data',
    'agg_main',
    'convert_criteria_pred_to_string',
    'convert_pred_to_prompt',
    'trialgpt_aggregation'
]