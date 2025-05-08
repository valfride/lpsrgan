import torch
from collections import Counter

def majority_vote_by_character(strings, slice_size=None):
    if slice_size==1:
        return strings[0]
    if slice_size is not None:
        strings = strings[0:slice_size]
    # Transpose list of strings to get a list of characters at each position
    transposed = zip(*strings)
    
    # For each position, find the most common character
    majority_vote = ''.join(Counter(chars).most_common(1)[0][0] for chars in transposed)
    
    return majority_vote

def select_highest_confidence_string(confidences, strings, slice_size=None):
    if slice_size==1:
        return strings[0]
    if slice_size is not None:
        strings = strings[0:slice_size]
        confidences = confidences[0:slice_size]
    
    # Convert confidence scores to a tensor if they aren't already
    if not isinstance(confidences, torch.Tensor):
        confidences = torch.tensor(confidences)
        
    # Find the index of the maximum confidence score
    max_conf_idx = torch.argmax(confidences)
    
    # Return the string corresponding to the highest confidence
    return strings[max_conf_idx]

def select_most_frequent_string(confidences, strings, slice_size=None):
    if slice_size==1:
        return strings[0]
    if slice_size is not None:
        strings = strings[0:slice_size]
        confidences = confidences[0:slice_size]
    
    # Ensure confidences are in list form, even if passed as a tensor
    if isinstance(confidences, torch.Tensor):
        confidences = confidences.tolist()
    
    # Count occurrences of each string
    string_counts = Counter(strings)
    # Find the maximum occurrence count
    max_count = max(string_counts.values())
    
    # Get all strings that have the maximum count (handle ties)
    most_frequent_strings = [s for s, count in string_counts.items() if count == max_count]
    
    if len(most_frequent_strings) == 1:
        # If only one string has the max count, return it
        return most_frequent_strings[0]
    else:
        # If there's a tie, select the one with the highest confidence
        # Find the index of the highest confidence among the tied strings
        max_conf_idx = max(
            (i for i, s in enumerate(strings) if s in most_frequent_strings),
            key=lambda i: confidences[i]
        )
        return strings[max_conf_idx]
