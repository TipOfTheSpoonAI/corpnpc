import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Union, Any, TYPE_CHECKING

# Import FAISS. If not available, provide a fallback warning.
try:
    import faiss
    # Type hinting for faiss.Index for static analysis.
    # TYPE_CHECKING makes this import only for type checkers, not runtime.
    if TYPE_CHECKING:
        from faiss import Index as FAISSIndex
except ImportError:
    faiss = None
    logger = logging.getLogger(__name__) # Ensure logger is available for the warning
    logger.warning(
        "FAISS library not found. Index loading and probing functionality will be unavailable. "
        "Please install it with 'pip install faiss-cpu' (or 'faiss-gpu')."
    )

# Import scikit-learn for vector normalization.
try:
    from sklearn.preprocessing import normalize
except ImportError:
    normalize = None
    logger = logging.getLogger(__name__) # Ensure logger is available for the warning
    logger.warning(
        "scikit-learn not found. Vector normalization functionality will be unavailable. "
        "Please install it with 'pip install scikit-learn'."
    )


# Set up logging for this module.
logger = logging.getLogger(__name__)


def load_faiss_index(index_path: str) -> "FAISSIndex":
    """
    Loads a FAISS index from a specified file path on disk.

    This function is responsible for deserializing a pre-built FAISS index
    from storage, making it ready for similarity search operations.

    Args:
        index_path: The file path (string) to the FAISS index file (e.g., 'vectors.faiss').

    Returns:
        A loaded FAISS index object (`faiss.Index`).

    Raises:
        FileNotFoundError: If the `index_path` does not exist.
        IsADirectoryError: If the `index_path` points to a directory instead of a file.
        RuntimeError: If FAISS library is not installed, or if the index file is corrupted
                      or cannot be read by FAISS.
        Exception: For any other unexpected errors during index loading.
    """
    if faiss is None:
        logger.error(
            "FAISS library not found. Cannot load index. "
            "Please ensure 'faiss-cpu' or 'faiss-gpu' is installed."
        )
        raise RuntimeError("FAISS is not installed.")

    index_file_path = Path(index_path)

    # --- Pre-checks for robustness ---
    if not index_file_path.exists():
        logger.error(f"FAISS index file not found: '{index_path}'.")
        raise FileNotFoundError(f"No such file or directory: '{index_path}'")
    if index_file_path.is_dir():
        logger.error(f"Provided path '{index_path}' is a directory, not a file. Cannot load FAISS index.")
        raise IsADirectoryError(f"'{index_path}' is a directory. Please provide a file path.")

    logger.info(f"Loading FAISS index from: '{index_path}'...")
    try:
        # faiss.read_index() handles the deserialization of the index.
        index = faiss.read_index(str(index_file_path))
        logger.info(f"FAISS index loaded successfully. Index type: {type(index).__name__}, "
                         f"Dimension: {index.d}, Number of vectors: {index.ntotal}")
        return index
    except RuntimeError as e:
        # FAISS often raises RuntimeError for internal errors like corrupted files
        logger.error(f"Failed to load FAISS index from '{index_path}': {e}. "
                     "The index file might be corrupted or incompatible.")
        raise # Re-raise for CLI to handle
    except Exception as e:
        logger.critical(f"An unexpected error occurred while loading FAISS index '{index_path}': {e}")
        raise


def probe_index(
    index: "FAISSIndex",
    embedded_corpus: List[Tuple[str, np.ndarray]]
) -> List[Tuple[int, float]]:
    """
    Probes a loaded FAISS index with a list of embedded corpus vectors to find
    the top-1 (nearest) match for each query embedding.

    The query embeddings are L2-normalized before searching to ensure that the
    distances computed by FAISS correspond to cosine similarity (assuming
    the FAISS index itself was built from L2-normalized vectors and is of
    an appropriate type like `IndexFlatIP` or `IndexFlatL2` with distance conversion).

    Args:
        index: The loaded FAISS index object (e.g., from `load_faiss_index`).
        embedded_corpus: A list of tuples, where each tuple is (original_line, embedding_vector).
                         The embedding_vector is a NumPy array representing the embedding.

    Returns:
        A list of tuples, where each tuple contains:
        - match_id (int): The integer ID of the nearest matching vector in the FAISS index.
        - cosine_similarity (float): The cosine similarity score (ranging from -1.0 to 1.0,
                                     or 0.0 to 1.0 for typical embeddings) for the top match.

    Raises:
        RuntimeError: If scikit-learn's `normalize` is not available or if FAISS is not installed.
        ValueError: If the `embedded_corpus` is empty or contains malformed embeddings.
        Exception: For any other unexpected errors during the probing process.
    """
    if normalize is None:
        logger.error(
            "scikit-learn's 'normalize' function not found. Cannot normalize embeddings. "
            "Please ensure 'scikit-learn' is installed."
        )
        raise RuntimeError("scikit-learn is not installed, cannot normalize vectors.")
    if faiss is None:
        logger.error(
            "FAISS library not found. Cannot probe index. "
            "Please ensure 'faiss-cpu' or 'faiss-gpu' is installed."
        )
        raise RuntimeError("FAISS is not installed.")

    if not embedded_corpus:
        logger.warning("Embedded corpus is empty. No probing will be performed.")
        return []

    # Extract only the embedding vectors for FAISS search.
    # Convert list of np.ndarray to a single 2D numpy array for efficient processing.
    try:
        query_vectors = np.array([vec for _, vec in embedded_corpus], dtype=np.float32)
        if query_vectors.ndim != 2:
            raise ValueError(f"Expected 2D array of embeddings, got {query_vectors.ndim} dimensions.")
        if query_vectors.shape[0] == 0:
            logger.warning("After extracting, no query vectors found. Returning empty results.")
            return []
    except Exception as e:
        logger.error(f"Error preparing query vectors from embedded corpus: {e}")
        raise ValueError(f"Invalid embedded corpus format: {e}")

    # L2-normalize the query embeddings.
    # This is crucial for calculating cosine similarity using Euclidean distance (L2) or Inner Product (IP).
    logger.info(f"Normalizing {query_vectors.shape[0]} query embeddings (L2 norm)...")
    normalized_query_vectors = normalize(query_vectors, axis=1, copy=False) # copy=False for efficiency

    # Perform the search. We're looking for the top-1 (nearest) match for each query vector.
    logger.info(f"Probing FAISS index with {normalized_query_vectors.shape[0]} embeddings...")
    try:
        # D = distances/similarities, I = indices/IDs of the matched vectors in the index
        distances, indices = index.search(normalized_query_vectors.astype(np.float32), 1)

        # Corrected: If IndexFlatIP is used with normalized vectors, 'd' IS the cosine similarity.
        # The previous '1 - d' would convert a 1.0 similarity to 0.0, causing filtering issues.
        results: List[Tuple[int, float]] = []
        for i, d in zip(indices.flatten().tolist(), distances.flatten().tolist()):
            if i != -1:
                # For IndexFlatIP with normalized vectors, 'd' is directly the cosine similarity.
                cosine_score = d # FIX APPLIED HERE
                results.append((i, float(cosine_score)))
            else:
                logger.debug("FAISS returned -1 index, no match found for a query vector.")

        logger.info(f"Probing complete. Found {len(results)} top-1 matches.")
        return results

    except Exception as e:
        logger.critical(f"An error occurred during FAISS probing: {e}")
        raise # Re-raise for CLI to handle
