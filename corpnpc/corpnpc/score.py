import logging
from typing import List, Tuple, Set, Union, Dict, Literal

# Set up logging for this module.
logger = logging.getLogger(__name__)

# Define a type alias for better readability and maintainability
# This describes the format of the output from this module.
# The 'Literal' type allows us to restrict string values to a predefined set.
MatchReportEntry = Tuple[str, int, float, Literal["near-exact", "strong", "likely", "weak", "filtered"]]


def _label_confidence(score: float) -> Literal["near-exact", "strong", "likely", "weak"]:
    """
    Helper function to categorize a cosine similarity score into a confidence label.
    """
    if score >= 0.98:
        return "near-exact"
    elif score >= 0.95:
        return "strong"
    elif score >= 0.90:
        return "likely"
    else:
        return "weak"


def filter_matches(
    raw_matches: List[Tuple[str, int, float]],
    threshold: float = 0.90,
    include_reasons: bool = False,
    return_dicts: bool = False,
) -> Union[List[Tuple[str, int, float]], List[Dict[str, Union[str, int, float]]]]:
    """
    Filters a list of raw similarity matches based on a cosine similarity threshold,
    ensures uniqueness by FAISS vector ID, and optionally adds confidence labels
    and skipped reasons.

    This function refines the raw output from `crack.py` to present only the
    most confident and distinct potential leak indicators, with enhanced
    metadata for better analysis and reporting.

    Args:
        raw_matches: A list of tuples, where each tuple represents a raw match from
                     the FAISS probing. Each tuple contains:
                     - original_line (str): The text line from the corpus that was embedded.
                     - vector_id (int): The integer ID of the matched vector in the FAISS index.
                     - cosine_similarity (float): The cosine similarity score for the match.
        threshold: The minimum cosine similarity score (between 0.0 and 1.0)
                   required for a match to be considered high-confidence. Matches
                   below this threshold will be discarded. Defaults to 0.90.
        include_reasons: If True, the returned tuples will include an additional
                         string indicating the reason for inclusion ("passed")
                         or exclusion (e.g., "below_threshold", "duplicate_id").
                         Note: Skipped reasons are primarily for auditing and debugging;
                         they will not be included if `return_dicts` is False.
        return_dicts: If True, the function returns a List of Dictionaries instead
                      of Tuples, providing named fields for clearer downstream processing.
                      If `include_reasons` is True, the reason will be an additional
                      key in the dictionary.

    Returns:
        A new list of tuples or dictionaries containing only the filtered,
        high-confidence, and unique matches. The format of the tuples is
        (original_line, vector_id, cosine_similarity, [reason_tag_optional]).
        The format of the dictionaries is `{"line": str, "vector_id": int,
        "similarity": float, "confidence": str, ["reason": str_optional]}`.

    Raises:
        ValueError: If the `threshold` is outside the valid range (0.0 to 1.0).
        TypeError: If `raw_matches` or its elements are not in the expected format.
    """
    if not (0.0 <= threshold <= 1.0):
        logger.error(f"Invalid threshold value: {threshold}. Threshold must be between 0.0 and 1.0.")
        raise ValueError("Threshold must be between 0.0 and 1.0.")

    if not raw_matches:
        logger.info("No raw matches provided for filtering. Returning empty list.")
        return []

    seen_vector_ids: Set[int] = set() # To track unique FAISS vector IDs
    processed_matches: List[Dict[str, Union[str, int, float]]] = [] # Always build dicts internally for flexibility

    logger.info(f"Filtering {len(raw_matches)} raw matches with threshold >= {threshold}...")

    for i, match in enumerate(raw_matches):
        current_reason: str = "passed"
        try:
            # Unpack the tuple. This will raise ValueError if not a 3-element tuple.
            original_line, vector_id, cosine_similarity = match

            # Validate types to ensure robust processing
            if not isinstance(original_line, str):
                current_reason = "malformed_original_line_type"
                logger.warning(f"Match {i}: Expected string for original_line, got {type(original_line)}. Skipping.")
                continue # Skip to next match
            if not isinstance(vector_id, int):
                current_reason = "malformed_vector_id_type"
                logger.warning(f"Match {i}: Expected int for vector_id, got {type(vector_id)}. Skipping.")
                continue # Skip to next match
            if not isinstance(cosine_similarity, (float, int)):
                current_reason = "malformed_similarity_type"
                logger.warning(f"Match {i}: Expected float/int for similarity, got {type(cosine_similarity)}. Skipping.")
                continue # Skip to next match

            # Apply the threshold filter.
            if cosine_similarity >= threshold:
                # Ensure uniqueness by FAISS vector ID.
                # The first time we see a vector ID that meets the threshold, we keep it.
                if vector_id not in seen_vector_ids:
                    confidence_label = _label_confidence(cosine_similarity)
                    match_data = {
                        "line": original_line,
                        "vector_id": vector_id,
                        "similarity": cosine_similarity,
                        "confidence": confidence_label,
                    }
                    if include_reasons:
                        match_data["reason"] = current_reason # Will be "passed"
                    processed_matches.append(match_data)
                    seen_vector_ids.add(vector_id)
                else:
                    current_reason = "duplicate_vector_id"
                    logger.debug(f"Skipping duplicate vector_id {vector_id} (score: {cosine_similarity:.4f}) for line: '{original_line[:50]}...'. Reason: {current_reason}")
            else:
                current_reason = "below_threshold"
                logger.debug(f"Skipping match below threshold (score: {cosine_similarity:.4f} < {threshold}) for line: '{original_line[:50]}...'. Reason: {current_reason}")

        except ValueError as e:
            current_reason = "malformed_tuple_structure"
            logger.warning(f"Match {i}: Malformed match tuple. Expected (str, int, float). Error: {e}. Skipping. Reason: {current_reason}")
            continue
        except Exception as e:
            current_reason = "unexpected_error"
            logger.error(f"An unexpected error occurred while processing match {i}: {e}. Skipping. Reason: {current_reason}")
            continue

    logger.info(f"Filtering complete. {len(processed_matches)} high-confidence, unique matches found.")

    if return_dicts:
        return processed_matches
    else:
        # Convert back to list of tuples if dicts are not requested.
        # This maintains the original API structure if desired.
        return [
            (d["line"], d["vector_id"], d["similarity"])
            for d in processed_matches
        ]

# No 'if __name__ == "__main__":' block for self-testing in production modules.
# Testing of this function should be done via a dedicated test suite (e.g., using pytest).