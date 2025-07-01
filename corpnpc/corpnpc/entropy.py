import logging
from collections import Counter
from typing import List, Tuple, Dict, Set, Union, Any # Added Union and Any
from math import log2

logger = logging.getLogger(__name__)

def calculate_ngram_frequencies(
    corpus_path: str,
    ngram_size: int = 1,
    lower_case: bool = True
) -> Counter:
    """
    Calculates the frequency of n-grams in a given text corpus.

    Args:
        corpus_path: Path to the plaintext .txt file.
        ngram_size: The size of the n-grams (e.g., 1 for unigrams, 2 for bigrams).
        lower_case: If True, converts all text to lower case before processing.

    Returns:
        A Counter object where keys are n-grams and values are their frequencies.
    """
    frequencies = Counter()
    total_ngrams = 0

    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                processed_line = line.strip()
                if lower_case:
                    processed_line = processed_line.lower()

                words = processed_line.split() # Simple split by space
                
                if ngram_size == 1:
                    for word in words:
                        frequencies[word] += 1
                        total_ngrams += 1
                else:
                    # Generate n-grams (simple word-based n-grams)
                    if len(words) >= ngram_size:
                        for i in range(len(words) - ngram_size + 1):
                            ngram = tuple(words[i : i + ngram_size])
                            frequencies[ngram] += 1
                            total_ngrams += 1
                
                if line_num % 10000 == 0:
                    logger.info(f"Processed {line_num} lines from {corpus_path} for n-gram frequencies...")

        logger.info(f"Completed n-gram frequency calculation for '{corpus_path}'. Found {len(frequencies)} unique n-grams.")
        if total_ngrams == 0:
            logger.warning(f"No n-grams found in '{corpus_path}'. Check corpus content or ngram_size.")
        return frequencies
    except FileNotFoundError:
        logger.error(f"Corpus file not found: '{corpus_path}'")
        raise
    except Exception as e:
        logger.critical(f"Error calculating n-gram frequencies for '{corpus_path}': {e}")
        raise


def calculate_entropy(frequencies: Counter, total_count: int) -> Dict[Union[str, Tuple[str, ...]], float]:
    """
    Calculates the Shannon entropy (or information content) for each n-gram.
    Lower entropy (higher probability) means less "surprising" or common.
    Higher entropy (lower probability) means more "surprising" or rare.

    Args:
        frequencies: A Counter object of n-gram frequencies.
        total_count: The total number of n-grams counted (sum of all frequencies).

    Returns:
        A dictionary mapping n-grams to their entropy scores (in bits).
    """
    if total_count == 0:
        logger.warning("Total count is zero. Cannot calculate entropy. Returning empty dict.")
        return {}
    
    entropy_scores = {}
    for ngram, count in frequencies.items():
        probability = count / total_count
        # Entropy is -log2(probability). Lower probability means higher entropy.
        entropy_scores[ngram] = -log2(probability)
    return entropy_scores


def find_rare_common_phrases(
    sensitive_corpus_path: str,
    background_corpus_path: str,
    ngram_size: int = 1,
    rarity_threshold_percentile: float = 20.0, # e.g., only consider phrases in bottom 20% of background frequency
    lower_case: bool = True
) -> List[Dict[str, Any]]:
    """
    Identifies phrases that appear in the sensitive corpus AND are rare (high entropy)
    in the general background corpus.

    Args:
        sensitive_corpus_path: Path to the sensitive plaintext .txt file.
        background_corpus_path: Path to a larger, general plaintext .txt file (e.g., LLM training data dump).
        ngram_size: The size of n-grams to consider.
        rarity_threshold_percentile: A percentile (0-100). Only n-grams with a frequency
                                     below this percentile in the background corpus will be
                                     considered "rare." Lower percentile means rarer.
        lower_case: If True, converts all text to lower case before processing.

    Returns:
        A list of dictionaries, each describing a rare common phrase:
        {
            "phrase": str | tuple,
            "sensitive_freq": int,
            "background_freq": int,
            "background_probability": float,
            "background_entropy_bits": float,
            "rarity_rank": int # Rank within the rare background phrases
        }
    """
    logger.info(f"Starting entropy scan for sensitive corpus '{sensitive_corpus_path}' against background '{background_corpus_path}'...")
    logger.info(f"N-gram size: {ngram_size}, Rarity Threshold Percentile: {rarity_threshold_percentile}%")

    sensitive_freqs = calculate_ngram_frequencies(sensitive_corpus_path, ngram_size, lower_case)
    background_freqs = calculate_ngram_frequencies(background_corpus_path, ngram_size, lower_case)

    if not sensitive_freqs or not background_freqs:
        logger.warning("One or both corpora are empty. No common rare phrases to find.")
        return []

    total_background_ngrams = sum(background_freqs.values())
    if total_background_ngrams == 0:
        logger.warning("Background corpus contains no n-grams. Cannot determine rarity.")
        return []

    background_probabilities = {
        ngram: count / total_background_ngrams
        for ngram, count in background_freqs.items()
    }
    
    background_entropy_scores = calculate_entropy(background_freqs, total_background_ngrams)

    # Determine rarity threshold based on background frequencies
    all_background_counts = sorted(background_freqs.values())
    if not all_background_counts: # Handle case where background_freqs is empty after filtering
        logger.warning("No valid n-gram counts in background corpus to determine rarity threshold.")
        return []

    # Calculate the frequency value corresponding to the rarity_threshold_percentile
    # np.percentile needs a non-empty array
    # Import numpy locally if it's only used here or pass it in. For a quick fix, let's add it.
    import numpy as np # Added numpy import within the function for _param_validation.py's internal numpy call

    if not all_background_counts:
        rarity_frequency_cutoff = 0
    else:
        # We want "rare," so we look at the lower end of frequencies.
        # A 20th percentile means values *below* or equal to this are rare.
        rarity_frequency_cutoff = np.percentile(all_background_counts, rarity_threshold_percentile)

    logger.info(f"Background corpus rarity frequency cutoff (below this is 'rare'): {rarity_frequency_cutoff}")

    rare_common_phrases_found = []
    
    # Track which phrases have been added to ensure uniqueness in output
    added_phrases_set = set()

    # Iterate through sensitive phrases and check against background
    for sensitive_ngram in sensitive_freqs.keys():
        if sensitive_ngram in background_freqs:
            background_count = background_freqs[sensitive_ngram]
            # Check if it's rare in the background corpus
            if background_count <= rarity_frequency_cutoff: # Using <= to include the exact percentile value
                if sensitive_ngram not in added_phrases_set: # Ensure uniqueness
                    phrase_str = " ".join(sensitive_ngram) if isinstance(sensitive_ngram, tuple) else sensitive_ngram
                    rare_common_phrases_found.append({
                        "phrase": phrase_str,
                        "sensitive_frequency": sensitive_freqs[sensitive_ngram],
                        "background_frequency": background_count,
                        "background_probability": background_probabilities.get(sensitive_ngram, 0.0),
                        "background_entropy_bits": background_entropy_scores.get(sensitive_ngram, 0.0),
                    })
                    added_phrases_set.add(sensitive_ngram)

    # Sort results by background entropy (higher entropy = rarer, more significant)
    rare_common_phrases_found.sort(key=lambda x: x["background_entropy_bits"], reverse=True)

    # Add a rarity rank for display
    for i, entry in enumerate(rare_common_phrases_found):
        entry["rarity_rank"] = i + 1

    logger.info(f"Found {len(rare_common_phrases_found)} rare phrases common to both corpora.")
    return rare_common_phrases_found
