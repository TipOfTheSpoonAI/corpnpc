import logging
import sys
import typer
from pathlib import Path
from typing import List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np # Needed for potential numpy array handling in CLI for new commands

# Import core functionalities from our CORPNPC modules
from corpnpc.embed import embed_corpus, get_embedder, BaseEmbedder
from corpnpc.crack import load_faiss_index, probe_index
from corpnpc.score import filter_matches
from corpnpc.report import save_leak_report

# New imports for fingerprinting and entropy scanning
from corpnpc.fingerprint import load_embeddings_from_index, reduce_dimensions, visualize_embeddings
from corpnpc.entropy import find_rare_common_phrases # No need for other entropy functions, as find_rare_common_phrases is the main entry point

# Initialize the Typer application
app = typer.Typer(
    name="corpnpc",
    help="CORPNPC: Compliance Oversight & Risk Probe for Neural Parrot Convergence. "
         "A forensic toolkit for detecting and analyzing data leakage in LLM embedding spaces.",
    add_completion=False
)

# --- Logging Setup ---
def setup_logging(verbose: bool = False, debug: bool = False):
    """
    Configures the logging system for the CORPNPC application.
    """
    log_level = logging.WARNING
    if verbose:
        log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.debug(f"Logging level set to {logging.getLevelName(log_level)}")


# --- Main CLI Command: 'crack' (Existing Leak Detection) ---
@app.command()
def crack(
    index: Path = typer.Option(
        ..., # '...' indicates this option is required
        "--index", "-i",
        help="Path to the pre-built FAISS index file (e.g., './vectors.faiss').",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        rich_help_panel="Crack Options"
    ),
    ngrams: Path = typer.Option(
        ..., # Required
        "--ngrams", "-n",
        help="Path to the plaintext .txt file containing the n-gram corpus (one n-gram per line).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        rich_help_panel="Crack Options"
    ),
    backend: str = typer.Option(
        "sentence-transformer",
        "--backend", "-b",
        help="Embedding backend to use. Options: 'sentence-transformer'.",
        rich_help_panel="Embedding Options"
    ),
    model_name: str = typer.Option(
        "all-MiniLM-L6-v2", # Default model for Sentence-Transformer
        "--model", "-m",
        help="Name of the specific model to use for the chosen embedding backend "
             "(e.g., 'all-MiniLM-L6-v2' for Sentence-Transformer).",
        rich_help_panel="Embedding Options"
    ),
    output: Path = typer.Option(
        Path("./corpnpc_crack_leak_report.json"), # Changed default name for NPC user clarity
        "--output", "-o",
        help="Path to save the leak detection report in JSON format.",
        resolve_path=True,
        rich_help_panel="Reporting Options"
    ),
    threshold: float = typer.Option(
        0.90, # Default cosine similarity threshold
        "--threshold", "-t",
        min=0.0, max=1.0, # Typer's built-in range validation
        help="Cosine similarity threshold (0.0 to 1.0). Matches with scores below "
             "this value will be filtered out. Higher values mean higher confidence leaks.",
        rich_help_panel="Crack Options"
    ),
    test_mode: bool = typer.Option(
        False,
        "--test-mode",
        help="Inject a known duplicate vector to validate matching, bypassing normal embedding. "
             "Requires the FAISS index to be accessible for fetching a known vector.",
        rich_help_panel="Development/Testing"
    ),
    bidirectional: bool = typer.Option(
        False,
        "--bidirectional",
        help="Enable bidirectional probing (A->B and B->A). Requires a second FAISS index and n-gram corpus for the reverse direction.",
        rich_help_panel="Crack Options"
    ),
    reverse_index: Optional[Path] = typer.Option(
        None,
        "--reverse-index", "-ri",
        help="Path to the FAISS index for the reverse probing direction (B->A). Required with --bidirectional.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        rich_help_panel="Crack Options"
    ),
    reverse_ngrams: Optional[Path] = typer.Option(
        None,
        "--reverse-ngrams", "-rn",
        help="Path to the n-gram corpus for the reverse probing direction (B->A). Required with --bidirectional.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        rich_help_panel="Crack Options"
    ),
    concurrent: bool = typer.Option(
        False,
        "--concurrent",
        help="Enable concurrent execution for bidirectional probing. Reduces total runtime for large datasets.",
        rich_help_panel="Performance Options"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output (INFO level logging).",
        rich_help_panel="Logging Options"
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug output (DEBUG level logging). Overrides --verbose.",
        rich_help_panel="Logging Options"
    )
):
    """
    Cracks a FAISS index to detect potential embedding leaks by probing it
    with a given n-gram corpus.

    The process involves embedding the n-grams, querying the FAISS index for
    similar vectors, filtering for high-confidence unique matches, and
    generating a detailed leak report.
    """
    # Record start time for overall performance tracking
    start_time = time.time()

    # 1. Setup Logging based on user's verbose/debug flags
    setup_logging(verbose=verbose, debug=debug)
    logger = logging.getLogger(__name__) # Get CLI-specific logger after setup

    typer.echo("--- CORPNPC: Compliance Oversight & Risk Probe (CRACK MODE) ---")
    typer.echo(f"  Primary FAISS Index: '{index}'")
    typer.echo(f"  Primary N-gram Corpus: '{ngrams}'")
    typer.echo(f"  Output Report: '{output}'")
    typer.echo(f"  Similarity Threshold: {threshold:.2f}")
    typer.echo(f"  Embedding Backend: '{backend}'")
    typer.echo(f"  Embedding Model: '{model_name}'")
    if test_mode:
        typer.echo("  Mode: TEST MODE (Bypassing normal embedding for deterministic test)")
    if bidirectional:
        typer.echo(f"  Mode: BIDIRECTIONAL PROBING (A->B and B->A)")
        typer.echo(f"    Reverse FAISS Index: '{reverse_index}'")
        typer.echo(f"    Reverse N-gram Corpus: '{reverse_ngrams}'")
        # Validate that reverse paths are provided if bidirectional is true
        if not reverse_index or not reverse_ngrams:
            logger.critical("Error: --reverse-index and --reverse-ngrams are required when --bidirectional is enabled.")
            typer.echo("Error: Bidirectional probing requires both --reverse-index and --reverse-ngrams. Please provide both paths.")
            raise typer.Exit(code=1)
        if concurrent: # Display concurrency status if bidirectional
            typer.echo(f"  Concurrency: Enabled (using ThreadPoolExecutor)")
        else:
            typer.echo(f"  Concurrency: Disabled (running sequentially)")
    typer.echo("-" * 70)

    # --- 2. Instantiate the Embedder (only if not in test_mode) ---
    current_embedder: Optional[BaseEmbedder] = None
    if not test_mode: # Only initialize embedder if we're doing real embedding
        try:
            typer.echo(f"Initializing embedding backend: '{backend}' with model: '{model_name}'...")
            embedder_init_start_time = time.time() # Start timing for embedder init
            current_embedder = get_embedder(backend=backend, model_name=model_name)
            embedder_init_end_time = time.time() # End timing
            logger.info(f"Successfully initialized embedder for backend '{backend}' in {embedder_init_end_time - embedder_init_start_time:.2f} seconds.")
        except (RuntimeError, ValueError, NotImplementedError) as e:
            logger.critical(f"Failed to initialize embedding backend: {e}")
            typer.echo(f"Error: Failed to initialize embedding backend. Check logs for details.")
            raise typer.Exit(code=1)
        except Exception as e:
            logger.critical(f"An unexpected error occurred during embedder initialization: {e}")
            typer.echo(f"Error: An unexpected error occurred. Check logs for details.")
            raise typer.Exit(code=1)
    else:
        logger.info("Skipping embedder initialization due to test_mode.")


    # --- Function to perform a single probe direction ---
    def perform_probe_direction(
        current_ngrams_path: Path,
        current_index_path: Path,
        direction_label: str
    ) -> List[Tuple[str, int, float]]:
        """
        Helper function to encapsulate the steps for one probing direction.
        """
        typer.echo(f"--- Probing Direction: {direction_label} ---")
        
        # Embed the N-gram Corpus (or inject test data)
        embedding_start_time = time.time() # Start timing for embedding
        embedded_lines: List[Tuple[str, np.ndarray]] = []
        try:
            typer.echo(f"Step A: Embedding n-gram corpus for {direction_label} from '{current_ngrams_path}'...")
            embedded_lines = embed_corpus(
                corpus_path=str(current_ngrams_path),
                embedder=current_embedder,
                index_path=str(current_index_path) if test_mode else None, # Pass index_path for test_mode
                test_mode=test_mode
            )
            if not embedded_lines:
                logger.warning(f"No lines were embedded for {direction_label}. Skipping this direction.")
                return []
            embedding_end_time = time.time() # End timing
            logger.info(f"Successfully embedded/generated {len(embedded_lines)} n-gram lines for {direction_label} in {embedding_end_time - embedding_start_time:.2f} seconds.")
        except Exception as e:
            logger.critical(f"Failed to embed n-gram corpus from '{current_ngrams_path}' for {direction_label}: {e}")
            typer.echo(f"Error: Failed to embed corpus for {direction_label}. Check logs for details.")
            raise typer.Exit(code=1)

        # Load the FAISS Index
        faiss_load_start_time = time.time() # Start timing for FAISS load
        faiss_index: Any = None
        try:
            typer.echo(f"Step B: Loading FAISS index for {direction_label} from '{current_index_path}'...")
            faiss_index = load_faiss_index(str(current_index_path))
            faiss_load_end_time = time.time() # End timing
            logger.info(f"Loaded FAISS index with {faiss_index.ntotal} vectors of dimension {faiss_index.d} for {direction_label} in {faiss_load_end_time - faiss_load_start_time:.2f} seconds.")
            
            # Basic dimensionality check between embeddings and index
            if embedded_lines and faiss_index.d != embedded_lines[0][1].shape[0]:
                logger.error(
                    f"Dimension mismatch for {direction_label}: Corpus embeddings ({embedded_lines[0][1].shape[0]}) "
                    f"do not match FAISS index dimension ({faiss_index.d}). "
                    "Ensure your FAISS index was built with embeddings from the same model, or check the `--model` parameter."
                )
                typer.echo(f"Error: Dimension mismatch for {direction_label} between corpus embeddings and FAISS index. "
                           f"Expected {faiss_index.d} dimensions, got {embedded_lines[0][1].shape[0]}. "
                           "Please ensure the FAISS index and the chosen embedding model are compatible.")
                raise typer.Exit(code=1)

        except Exception as e:
            logger.critical(f"Failed to load FAISS index from '{current_index_path}' for {direction_label}: {e}")
            typer.echo(f"Error: Failed to load FAISS index for {direction_label}. Check logs for details.")
            raise typer.Exit(code=1)

        # Probe the FAISS Index
        probing_start_time = time.time() # Start timing for probing
        raw_matches_from_probe: List[Tuple[int, float]] = []
        try:
            typer.echo(f"Step C: Probing FAISS index for similarity matches for {direction_label}...")
            raw_matches_from_probe = probe_index(faiss_index, embedded_lines)
            if not raw_matches_from_probe:
                logger.warning(f"No matches found during FAISS probing for {direction_label}.")
                return [] # Return empty if no matches
            probing_end_time = time.time() # End timing
            logger.info(f"Found {len(raw_matches_from_probe)} raw matches from FAISS for {direction_label} in {probing_end_time - probing_start_time:.2f} seconds.")
        except Exception as e:
            logger.critical(f"Failed to probe FAISS index for {direction_label}: {e}")
            typer.echo(f"Error: Failed during FAISS probing for {direction_label}. Check logs for details.")
            raise typer.Exit(code=1)

        # Combine original lines with their probe results
        combined_raw_matches: List[Tuple[str, int, float]] = []
        for (original_line, _), (match_id, score) in zip(embedded_lines, raw_matches_from_probe):
            combined_raw_matches.append((original_line, match_id, score))
        
        typer.echo(f"--- Finished Probing Direction: {direction_label} ---")
        return combined_raw_matches

    # --- Main Probing Logic ---
    all_filtered_matches: List[Tuple[str, int, float]] = []

    if bidirectional and concurrent: # Bidirectional AND concurrent
        typer.echo("\n" + "="*70)
        typer.echo("Initiating concurrent bidirectional probing...")
        with ThreadPoolExecutor(max_workers=2) as executor: # Max 2 workers for two directions
            # Submit tasks for both directions
            future_ab = executor.submit(perform_probe_direction, ngrams, index, "A -> B")
            future_ba = executor.submit(perform_probe_direction, reverse_ngrams, reverse_index, "B -> A")

            try: # Error handling for concurrent results
                main_direction_matches = future_ab.result()
                reverse_direction_matches = future_ba.result()
            except Exception as e:
                logger.critical(f"Error in concurrent probing: {e}")
                typer.echo(f"Error: Bidirectional probing failed. Check logs for details.")
                raise typer.Exit(code=1) # Exit with an error code

        typer.echo("Concurrent probing complete. Proceeding with filtering.")
        typer.echo("="*70 + "\n")

        # Filter matches for A -> B
        typer.echo(f"Filtering {len(main_direction_matches)} matches for A->B with threshold >= {threshold:.2f}...")
        try:
            filtered_ab_matches = filter_matches(main_direction_matches, threshold=threshold)
            logger.info(f"Filtered down to {len(filtered_ab_matches)} high-confidence, unique potential leaks (A->B).")
            all_filtered_matches.extend(filtered_ab_matches)
        except Exception as e:
            logger.critical(f"Failed to filter A->B matches: {e}")
            typer.echo(f"Error: Failed during A->B match filtering. Check logs for details.")
            raise typer.Exit(code=1)

        # Filter matches for B -> A
        typer.echo(f"Filtering {len(reverse_direction_matches)} matches for B->A with threshold >= {threshold:.2f}...")
        try:
            filtered_ba_matches = filter_matches(reverse_direction_matches, threshold=threshold)
            logger.info(f"Filtered down to {len(filtered_ba_matches)} high-confidence, unique potential leaks (B->A).")
            all_filtered_matches.extend(filtered_ba_matches)
        except Exception as e:
            logger.critical(f"Failed to filter B->A matches: {e}")
            typer.echo(f"Error: Failed during B->A match filtering. Check logs for details.")
            raise typer.Exit(code=1)

    elif bidirectional and not concurrent: # Bidirectional AND sequential
        typer.echo("\n" + "="*70)
        typer.echo("Initiating sequential bidirectional probing...")
        
        main_direction_matches = perform_probe_direction(ngrams, index, "A -> B")
        reverse_direction_matches = perform_probe_direction(reverse_ngrams, reverse_index, "B -> A")

        typer.echo("Sequential probing complete. Proceeding with filtering.")
        typer.echo("="*70 + "\n")

        # Filter matches for A -> B
        typer.echo(f"Filtering {len(main_direction_matches)} matches for A->B with threshold >= {threshold:.2f}...")
        try:
            filtered_ab_matches = filter_matches(main_direction_matches, threshold=threshold)
            logger.info(f"Filtered down to {len(filtered_ab_matches)} high-confidence, unique potential leaks (A->B).")
            all_filtered_matches.extend(filtered_ab_matches)
        except Exception as e:
            logger.critical(f"Failed to filter A->B matches: {e}")
            typer.echo(f"Error: Failed during A->B match filtering. Check logs for details.")
            raise typer.Exit(code=1)

        # Filter matches for B -> A
        typer.echo(f"Filtering {len(reverse_direction_matches)} matches for B->A with threshold >= {threshold:.2f}...")
        try:
            filtered_ba_matches = filter_matches(reverse_direction_matches, threshold=threshold)
            logger.info(f"Filtered down to {len(filtered_ba_matches)} high-confidence, unique potential leaks (B->A).")
            all_filtered_matches.extend(filtered_ba_matches)
        except Exception as e:
            logger.critical(f"Failed to filter B->A matches: {e}")
            typer.echo(f"Error: Failed during B->A match filtering. Check logs for details.")
            raise typer.Exit(code=1)

    else: # Original single direction probing (not bidirectional)
        main_direction_matches = perform_probe_direction(ngrams, index, "A -> B")
        # Filter matches for A -> B
        typer.echo(f"Filtering {len(main_direction_matches)} matches for A->B with threshold >= {threshold:.2f}...")
        try:
            filtered_ab_matches = filter_matches(main_direction_matches, threshold=threshold)
            logger.info(f"Filtered down to {len(filtered_ab_matches)} high-confidence, unique potential leaks (A->B).")
            all_filtered_matches.extend(filtered_ab_matches)
        except Exception as e:
            logger.critical(f"Failed to filter A->B matches: {e}")
            typer.echo(f"Error: Failed during A->B match filtering. Check logs for details.")
            raise typer.Exit(code=1)

    # --- Final Filtering Check (consolidated for all scenarios) ---
    if not all_filtered_matches:
        logger.info("After all probing and filtering, no high-confidence leaks found.")
        typer.echo("No high-confidence potential leaks detected after all scans.")
        # Calculate and log elapsed time even for successful exit with no leaks
        end_time = time.time()
        elapsed_time = end_time - start_time
        typer.echo(f"\nTotal processing completed in {elapsed_time:.2f} seconds.")
        logger.info(f"CORPNPC operation completed successfully in {elapsed_time:.2f} seconds.")
        raise typer.Exit(code=0)


    # --- Save Leak Report ---
    try:
        typer.echo("Saving crack report...") # Specific to crack command
        # For crack command, we don't pass extensive metadata like faiss_origin_info yet.
        # This will be handled in report.py once it's extended.
        save_leak_report(all_filtered_matches, str(output), report_type="crack") # Pass report type
        logger.info(f"Crack report successfully saved to '{output.resolve()}'.")
    except Exception as e:
        logger.critical(f"Failed to save crack report to '{output}': {e}")
        typer.echo(f"Error: Failed to save crack report. Check logs for details.")
        raise typer.Exit(code=1)

    # --- Final Summary ---
    typer.echo("-" * 70)
    typer.echo("CORPNPC Crack Scan Complete!")
    typer.echo(f"  Report saved to: '{output.resolve()}'")
    typer.echo(f"  {len(all_filtered_matches)} total potential unique embedding leaks identified across all directions.")
    typer.echo("Review the report for detailed findings.")
    # Add final processing time to the summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    typer.echo(f"Total processing completed in {elapsed_time:.2f} seconds.")
    typer.echo("-" * 70)
    logger.info(f"CORPNPC crack operation completed successfully in {elapsed_time:.2f} seconds.")
    raise typer.Exit(code=0)


# --- New CLI Command: 'fingerprint' (for Embedding Distribution Analysis) ---
@app.command()
def fingerprint(
    index_paths: List[Path] = typer.Option(
        ...,
        "--index", "-i",
        help="Path(s) to the FAISS index files to fingerprint. Can specify multiple times.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        rich_help_panel="Fingerprint Options"
    ),
    output: Path = typer.Option(
        Path("./corpnpc_fingerprint_plot.png"), # Default output for fingerprint
        "--output", "-o",
        help="Path to save the embedding fingerprint plot (e.g., '.png', '.svg').",
        resolve_path=True,
        rich_help_panel="Reporting Options"
    ),
    sample_size: Optional[int] = typer.Option(
        10000,
        "--sample-size", "-s",
        help="Number of vectors to sample from each index for dimensionality reduction and plotting. "
             "Set to 0 or None to attempt to load all (memory intensive for large indexes).",
        min=0,
        rich_help_panel="Fingerprint Options"
    ),
    reduction_method: str = typer.Option(
        "umap", "-r",
        help="Dimensionality reduction method: 'umap' (default, requires umap-learn) or 'pca'.",
        rich_help_panel="Fingerprint Options"
    ),
    n_components: int = typer.Option(
        2,
        "--n-components", "-c",
        help="Number of dimensions to reduce to (typically 2 or 3 for visualization).",
        min=2, max=3,
        rich_help_panel="Fingerprint Options"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output (INFO level logging).",
        rich_help_panel="Logging Options"
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Enable debug output (DEBUG level logging). Overrides --verbose.",
        rich_help_panel="Logging Options"
    )
):
    """
    Analyzes and visualizes the geometric distribution of embeddings from one or more FAISS indexes.
    Useful for detecting "shadow clones" or models with highly similar underlying structures.
    """
    start_time = time.time()
    setup_logging(verbose=verbose, debug=debug)
    logger = logging.getLogger(__name__)

    typer.echo("--- CORPNPC: Embedding Distribution Fingerprinting ---")
    typer.echo(f"  FAISS Indexes to analyze: {[str(p) for p in index_paths]}")
    typer.echo(f"  Output Plot: '{output}'")
    typer.echo(f"  Sample Size per Index: {sample_size if sample_size else 'All'}")
    typer.echo(f"  Reduction Method: {reduction_method.upper()}")
    typer.echo(f"  Target Dimensions: {n_components}")
    typer.echo("-" * 70)

    if not index_paths:
        typer.echo("Error: At least one FAISS index path must be provided.")
        raise typer.Exit(code=1)
    
    # Load and reduce embeddings for each index
    reduced_embeddings_for_plot: List[Tuple[np.ndarray, str]] = []
    for i_path in index_paths:
        try:
            embeddings = load_embeddings_from_index(str(i_path), sample_size if sample_size > 0 else None)
            if embeddings.shape[0] == 0:
                logger.warning(f"No embeddings loaded from '{i_path}'. Skipping reduction for this index.")
                continue

            reduced = reduce_dimensions(embeddings, method=reduction_method, n_components=n_components)
            reduced_embeddings_for_plot.append((reduced, i_path.name))
        except Exception as e:
            logger.critical(f"Error processing index '{i_path}' for fingerprinting: {e}")
            typer.echo(f"Error: Failed to process '{i_path}'. Check logs for details.")
            raise typer.Exit(code=1)

    if not reduced_embeddings_for_plot:
        logger.info("No embeddings were successfully processed for visualization. Exiting.")
        typer.echo("No embeddings were successfully processed for visualization. No plot generated.")
        raise typer.Exit(code=0)

    # Visualize the reduced embeddings
    try:
        typer.echo(f"Generating and saving fingerprint plot to '{output}'...")
        visualize_embeddings(
            reduced_embeddings_for_plot,
            output_path=str(output),
            title="Embedding Distribution Fingerprint",
            xlabel=f"{reduction_method.upper()} Component 1",
            ylabel=f"{reduction_method.upper()} Component 2"
        )
        logger.info(f"Fingerprint plot successfully saved to '{output.resolve()}'.")
    except Exception as e:
        logger.critical(f"Failed to generate/save fingerprint plot: {e}")
        typer.echo(f"Error: Failed to generate/save plot. Check logs for details.")
        raise typer.Exit(code=1)

    end_time = time.time()
    elapsed_time = end_time - start_time
    typer.echo("-" * 70)
    typer.echo("CORPNPC Fingerprinting Complete!")
    typer.echo(f"  Plot saved to: '{output.resolve()}'")
    typer.echo(f"Total processing completed in {elapsed_time:.2f} seconds.")
    typer.echo("-" * 70)
    logger.info(f"CORPNPC fingerprint operation completed successfully in {elapsed_time:.2f} seconds.")
    raise typer.Exit(code=0)


# --- New CLI Command: 'entropy-scan' (for Rare Phrase Detection) ---
@app.command()
def entropy_scan(
    sensitive_ngrams: Path = typer.Option(
        ...,
        "--sensitive-ngrams", "-s",
        help="Path to the plaintext .txt file containing the sensitive n-gram corpus (one n-gram per line).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        rich_help_panel="Entropy Scan Options"
    ),
    background_corpus: Path = typer.Option(
        ...,
        "--background-corpus", "-b",
        help="Path to a larger, general plaintext .txt file representing the background data "
             "(e.g., a dump of LLM training data or public corpus).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        rich_help_panel="Entropy Scan Options"
    ),
    output: Path = typer.Option(
        Path("./corpnpc_entropy_report.json"), # Default output for entropy scan
        "--output", "-o",
        help="Path to save the entropy scan report in JSON format.",
        resolve_path=True,
        rich_help_panel="Reporting Options"
    ),
    ngram_size: int = typer.Option(
        1,
        "--ngram-size", "-n",
        help="The size of n-grams to consider (e.g., 1 for unigrams, 2 for bigrams).",
        min=1,
        rich_help_panel="Entropy Scan Options"
    ),
    rarity_percentile: float = typer.Option(
        20.0,
        "--rarity-percentile", "-r",
        min=0.0, max=100.0,
        help="A percentile (0-100). Only n-grams with a frequency below this percentile "
             "in the background corpus will be considered 'rare' and potentially leaked. "
             "Lower values mean rarer phrases.",
        rich_help_panel="Entropy Scan Options"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output (INFO level logging).",
        rich_help_panel="Logging Options"
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Enable debug output (DEBUG level logging). Overrides --verbose.",
        rich_help_panel="Logging Options"
    )
):
    """
    Identifies phrases that appear in a sensitive corpus AND are statistically rare
    (high entropy) in a larger background corpus. Useful for detecting leaked proprietary terms.
    """
    start_time = time.time()
    setup_logging(verbose=verbose, debug=debug)
    logger = logging.getLogger(__name__)

    typer.echo("--- CORPNPC: Entropy Scan for Rare Phrase Detection ---")
    typer.echo(f"  Sensitive N-grams: '{sensitive_ngrams}'")
    typer.echo(f"  Background Corpus: '{background_corpus}'")
    typer.echo(f"  Output Report: '{output}'")
    typer.echo(f"  N-gram Size: {ngram_size}")
    typer.echo(f"  Rarity Threshold Percentile: {rarity_percentile:.2f}%")
    typer.echo("-" * 70)

    try:
        rare_phrases = find_rare_common_phrases(
            sensitive_corpus_path=str(sensitive_ngrams),
            background_corpus_path=str(background_corpus),
            ngram_size=ngram_size,
            rarity_threshold_percentile=rarity_percentile
        )
        if not rare_phrases:
            logger.info("No rare common phrases found between the sensitive and background corpora.")
            typer.echo("No rare common phrases detected.")
            end_time = time.time()
            elapsed_time = end_time - start_time
            typer.echo(f"\nTotal processing completed in {elapsed_time:.2f} seconds.")
            logger.info(f"CORPNPC entropy scan operation completed successfully in {elapsed_time:.2f} seconds.")
            raise typer.Exit(code=0)

        # Save the report
        save_leak_report(rare_phrases, str(output), report_type="entropy") # Pass report type
        logger.info(f"Entropy scan report successfully saved to '{output.resolve()}'.")

    except Exception as e:
        logger.critical(f"Error during entropy scan: {e}")
        typer.echo(f"Error: Entropy scan failed. Check logs for details.")
        raise typer.Exit(code=1)

    end_time = time.time()
    elapsed_time = end_time - start_time
    typer.echo("-" * 70)
    typer.echo("CORPNPC Entropy Scan Complete!")
    typer.echo(f"  Report saved to: '{output.resolve()}'")
    typer.echo(f"  {len(rare_phrases)} rare common phrases identified.")
    typer.echo(f"Total processing completed in {elapsed_time:.2f} seconds.")
    typer.echo("-" * 70)
    logger.info(f"CORPNPC entropy scan operation completed successfully in {elapsed_time:.2f} seconds.")
    raise typer.Exit(code=0)


# Main entry point for the Typer application
if __name__ == "__main__":
    app()
