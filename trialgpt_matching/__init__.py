# trialgpt_matching/__init__.py
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .TrialGPT import (
    parse_criteria,
    print_trial,
    get_matching_prompt,
    trialgpt_match
)

from .run_matching import (
    parse_arguments,
    main
)

__all__ = [
    'parse_criteria',
    'print_trial',
    'get_matching_prompt',
    'trialgpt_match',
    'parse_arguments',
    'main'
]