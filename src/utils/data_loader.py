"""
Data loading utilities for speculative decoding benchmarks
"""

import json
from typing import List, Dict, Any


def json_loader(file_path: str) -> List[Dict[str, str]]:
    """
    Load prompts from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing prompts.
        
    Returns:
        A list of prompts to be used for inference.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data