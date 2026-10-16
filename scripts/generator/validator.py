import ast
import re
from typing import List, Dict, Tuple, Set

VALID_CATEGORIES = {
    'python_coding', 'backend_coding', 'ml_coding', 'rag_coding',
    'agent_coding', 'deep_learning_coding', 'computer_vision_coding',
    'code_review_coding', 'software_engineering_coding',
    'machine_learning', 'deep_learning', 'computer_vision',
    'llm_rag', 'ai_agents', 'python', 'software_engineering'
}

def tokenize(text: str) -> Set[str]:
    # Lowercase and extract alphanumeric words of length >= 3
    words = re.findall(r'[a-zA-Z0-9_]{3,}', text.lower())
    return set(words)

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def validate_python_syntax(code_str: str) -> Tuple[bool, str]:
    try:
        ast.parse(code_str)
        return True, ''
    except SyntaxError as e:
        return False, f'SyntaxError: {e.msg} at line {e.lineno}'

def validate_dataset(examples: List[Dict], existing_instructions: Set[str] = None) -> Dict:
    results = {
        'total': len(examples),
        'with_code': 0,
        'without_code': 0,
        'invalid_items': [],
        'exact_duplicates': [],
        'near_duplicates': [],
        'category_counts': {},
        'syntax_errors': []
    }
    
    seen_instructions = {}
    existing_tokens = {}
    
    if existing_instructions:
        for inst in existing_instructions:
            norm = ' '.join(inst.lower().split())
            seen_instructions[norm] = inst
            existing_tokens[inst] = tokenize(inst)
            
    for idx, item in enumerate(examples):
        if not isinstance(item, dict):
            results['invalid_items'].append({'index': idx, 'error': 'Item is not a dict'})
            continue
            
        cat = item.get('category')
        inst = item.get('instruction')
        resp = item.get('response')
        code = item.get('code')
        
        # Validate fields
        if not cat or not isinstance(cat, str) or not cat.strip():
            results['invalid_items'].append({'index': idx, 'error': 'Missing or invalid category'})
            continue
            
        if cat not in VALID_CATEGORIES:
            results['invalid_items'].append({'index': idx, 'error': f'Unknown category: {cat}'})
            
        if not inst or not isinstance(inst, str) or not inst.strip():
            results['invalid_items'].append({'index': idx, 'error': 'Missing or invalid instruction'})
            continue
            
        if not resp or not isinstance(resp, str) or not resp.strip():
            results['invalid_items'].append({'index': idx, 'error': 'Missing or invalid response'})
            continue
            
        # Track category
        results['category_counts'][cat] = results['category_counts'].get(cat, 0) + 1
        
        # Code presence
        if code and isinstance(code, str) and code.strip():
            results['with_code'] += 1
        else:
            results['without_code'] += 1
            
        # Deduplication checks
        norm_inst = ' '.join(inst.lower().split())
        if norm_inst in seen_instructions:
            results['exact_duplicates'].append({
                'index': idx,
                'instruction': inst,
                'matched_with': seen_instructions[norm_inst]
            })
        else:
            seen_instructions[norm_inst] = inst
            
        curr_tokens = tokenize(inst)
        for prev_inst, prev_tokens in existing_tokens.items():
            if norm_inst != ' '.join(prev_inst.lower().split()):
                sim = jaccard_similarity(curr_tokens, prev_tokens)
                if sim >= 0.85:
                    results['near_duplicates'].append({
                        'index': idx,
                        'similarity': sim,
                        'instruction': inst,
                        'matched_with': prev_inst
                    })
        existing_tokens[inst] = curr_tokens

    return results
