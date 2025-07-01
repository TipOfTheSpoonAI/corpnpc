import json
import logging
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Union, Literal, Any
from datetime import datetime

# Set up logging for this module.
logger = logging.getLogger(__name__)

# Type alias for clarity: defines the structure of a single report entry.
# This module is designed to accept either the original tuple format (from score.py)
# or the dictionary format (e.g., if score.py's return_dicts=True, or from entropy.py).
# The internal processing will convert to a consistent dictionary format.
ReportItem = Union[Tuple[str, int, float], Dict[str, Union[str, int, float, Literal["near-exact", "strong", "likely", "weak"]]]]
EntropyReportItem = Dict[str, Any] # Specific type for entropy scan results


def _hash_report_payload(data: dict) -> str:
    """
    Generates a SHA256 hash of the JSON representation of the report content.

    The data is first serialized to a JSON string with sorted keys to ensure
    consistent hashing regardless of dictionary iteration order. This hash
    provides a verifiable integrity check for the report.

    Args:
        data: The dictionary containing the full report content (including metadata and results).

    Returns:
        A hexadecimal string representing the SHA256 hash of the report data.
    """
    try:
        # json.dumps with sort_keys=True ensures consistent order for hashing.
        # .encode('utf-8') converts the string to bytes, which hashlib requires.
        json_string = json.dumps(data, sort_keys=True).encode('utf-8')
        return hashlib.sha256(json_string).hexdigest()
    except Exception as e:
        logger.error(f"Failed to serialize data for hashing: {e}")
        raise


def save_leak_report(
    results: Union[List[ReportItem], List[EntropyReportItem]],
    path: str,
    output_format: Literal['json', 'jsonl'] = 'json', # 'json' for single file, 'jsonl' for NDJSON
    include_hash: bool = False,                      # Whether to include SHA256 hash of report
    to_stdout: bool = False,                         # Whether to print report to stdout
    dry_run: bool = False,                           # If True, no files will be written to disk
    report_type: Literal['crack', 'entropy'] = 'crack' # New: Type of report being generated
) -> None:
    """
    Saves the leak detection results or entropy scan results to a JSON or NDJSON file,
    and optionally prints it to standard output.

    Each entry in the results list is transformed into a dictionary with
    appropriate fields based on the report_type. The report also includes
    metadata like generation timestamp and total count. Optional SHA256 hashing
    provides integrity verification.

    Args:
        results: A list of tuples/dictionaries for 'crack' reports, or
                 a list of dictionaries for 'entropy' reports.
        path: The base file path (string) where the report will be saved.
              A file extension will be automatically appended based on `output_format`.
        output_format: Specifies the output file format.
                       'json': Standard single JSON array file.
                       'jsonl': Newline-delimited JSON (NDJSON), one JSON object per line.
        include_hash: If True, a SHA256 hash of the full report payload will be
                      included as a top-level field in the JSON report for integrity verification.
        to_stdout: If True, the generated report content will also be printed
                   to standard output (console). Useful for CI/CD or quick debugging.
        dry_run: If True, the function will simulate the process but will NOT
                 write any files to disk. It will still print to stdout if `to_stdout` is True.
        report_type: Specifies the type of report: 'crack' for embedding leakage,
                     'entropy' for rare phrase detection.

    Raises:
        OSError: If there's an issue creating the output directory or writing the file
                 (only if `dry_run` is False).
        ValueError: If `output_format` is invalid or other input validation fails.
        Exception: For any other unexpected errors during JSON serialization or file operations.
    """
    if output_format not in ['json', 'jsonl']:
        logger.error(f"Invalid output format specified: '{output_format}'. Must be 'json' or 'jsonl'.")
        raise ValueError("Invalid output format. Must be 'json' or 'jsonl'.")
    
    if report_type not in ['crack', 'entropy']:
        logger.error(f"Invalid report type specified: '{report_type}'. Must be 'crack' or 'entropy'.")
        raise ValueError("Invalid report type. Must be 'crack' or 'entropy'.")

    # --- Path Management and Suffix Safeguard ---
    output_path = Path(path)
    expected_suffix = f".{output_format}"
    if output_path.suffix != expected_suffix:
        original_path = output_path
        output_path = output_path.with_suffix(expected_suffix)
        logger.info(f"Corrected output path from '{original_path}' to '{output_path}' (added/changed suffix to '{expected_suffix}').")

    if not dry_run and not output_path.parent.exists():
        logger.info(f"Creating parent directory for report: '{output_path.parent}' (if it does not exist).")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory '{output_path.parent}' for report output: {e}")
            raise

    payload_list: List[Dict[str, Any]] = []
    logger.info(f"Preparing {len(results)} results for export for '{report_type}' report...")

    # Iterate through the results and structure them into dictionaries based on report_type.
    for i, item in enumerate(results):
        try:
            processed_item: Dict[str, Any] = {}
            if report_type == 'crack':
                if isinstance(item, dict):
                    # If item is already a dictionary (from score.py's new output with return_dicts=True)
                    processed_item = {
                        "matched_text": item.get("line", ""),
                        "vector_id": item.get("vector_id", -1),
                        "similarity": round(float(item.get("similarity", 0.0)), 4),
                    }
                    if "confidence" in item:
                        processed_item["confidence_label"] = item["confidence"]
                    if "reason" in item:
                        processed_item["filtering_reason"] = item["reason"]
                elif isinstance(item, tuple) and len(item) >= 3:
                    # If item is the original tuple format (original_line, vector_id, cosine_similarity)
                    line, idx, score, *rest = item
                    processed_item = {
                        "matched_text": line,
                        "vector_id": idx,
                        "similarity": round(float(score), 4),
                    }
                    if len(rest) > 0 and isinstance(rest[0], str): # Check if reason might be present
                        # A simple check for known reason strings for clarity
                        if rest[0] in ["passed", "below_threshold", "duplicate_vector_id", "malformed_tuple_structure", "unexpected_error"]:
                            processed_item["filtering_reason"] = rest[0]
                    if len(rest) > 1 and isinstance(rest[1], str) and rest[1] in ["near-exact", "strong", "likely", "weak"]:
                        processed_item["confidence_label"] = rest[1]
                else:
                    logger.warning(f"Crack report item {i} has unexpected format: {item}. Expected tuple or dict. Skipping.")
                    continue
            
            elif report_type == 'entropy':
                # Entropy report items are expected to be dictionaries directly
                if isinstance(item, dict):
                    # Validate required fields for entropy report
                    if all(key in item for key in ["phrase", "sensitive_frequency", "background_frequency", "background_probability", "background_entropy_bits", "rarity_rank"]):
                        processed_item = item
                    else:
                        logger.warning(f"Entropy report item {i} missing required keys: {item}. Skipping.")
                        continue
                else:
                    logger.warning(f"Entropy report item {i} has unexpected format: {item}. Expected dict. Skipping.")
                    continue

            # Final validation of critical fields after processing (basic check for non-empty)
            if not processed_item:
                logger.warning(f"Result item {i} processed into empty dictionary. Skipping.")
                continue

            payload_list.append(processed_item)
        except Exception as e:
            logger.error(f"Error processing result item {i} ({item}) for report type '{report_type}': {e}. Skipping this item.")
            continue

    # Construct the full report content, including metadata.
    report_content: Dict[str, Any] = {
        "report_generated_at": datetime.now().isoformat(),
        f"total_{report_type}_results_reported": len(payload_list),
    }

    # Add the actual results list with a type-specific key
    if report_type == 'crack':
        report_content["embedding_leak_detection_results"] = payload_list
    elif report_type == 'entropy':
        report_content["rare_phrase_entropy_scan_results"] = payload_list

    # Add SHA256 hash if requested.
    if include_hash:
        try:
            report_content["report_sha256_hash"] = _hash_report_payload(report_content)
            logger.info("SHA256 hash included in the report for integrity verification.")
        except Exception as e:
            logger.error(f"Failed to generate SHA256 hash for report: {e}. Proceeding without hash.")
            # Do not raise, as hash is an optional feature for report generation.

    # --- Write the structured data to the file (if not dry_run) ---
    if not dry_run:
        try:
            if output_format == 'json':
                logger.info(f"Saving '{report_type}' report in JSON format to: '{output_path}'...")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(report_content, f, indent=2) # indent=2 for human-readable JSON
            elif output_format == 'jsonl':
                logger.info(f"Saving '{report_type}' report in NDJSON (JSON Lines) format to: '{output_path}'...")
                with open(output_path, "w", encoding="utf-8") as f:
                    # For NDJSON, only the individual records are written, one per line.
                    # Top-level metadata like timestamp, total count, and hash are omitted from the file.
                    for entry in payload_list:
                        f.write(json.dumps(entry) + "\n")
                logger.warning(
                    "When using 'jsonl' format, top-level metadata (timestamp, total count, hash) "
                    "is NOT included in the .jsonl file itself, as it violates the JSON Lines spec. "
                    "Only individual records are written."
                )

            logger.info(f"'{report_type}' report successfully saved to: '{output_path.resolve()}'")

        except OSError as e:
            logger.error(f"Failed to write '{report_type}' report to '{output_path}': {e}")
            raise
        except Exception as e:
            logger.critical(f"An unexpected error occurred while saving the '{report_type}' report: {e}")
            raise
    else:
        logger.info(f"Dry run enabled. '{report_type}' report NOT written to disk: '{output_path}'.")

    # --- Optional Output to STDOUT ---
    if to_stdout:
        logger.info(f"Printing '{report_type}' report to standard output...")
        try:
            if output_format == 'json':
                # For 'json' format, print the full report content (including metadata and hash).
                print(json.dumps(report_content, indent=2))
            elif output_format == 'jsonl':
                # For 'jsonl' format, print each individual record to stdout, one per line.
                for entry in payload_list:
                    print(json.dumps(entry)) # No indent for simple JSONL stdout

            # Emit hash to stdout if requested AND printing to stdout
            if include_hash and "report_sha256_hash" in report_content:
                print(f"\nReport SHA256 Hash: {report_content['report_sha256_hash']}")

        except Exception as e:
            logger.error(f"Failed to print '{report_type}' report to standard output: {e}")
            # Do not raise, as printing to stdout is a secondary function.

# No 'if __name__ == "__main__":' block for self-testing in production modules.
# Testing should be done via a dedicated test suite (e.g., using pytest).
