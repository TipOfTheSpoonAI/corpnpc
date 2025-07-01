import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Any, Optional, TYPE_CHECKING
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Try importing UMAP, provide fallback if not installed
try:
    import umap
except ImportError:
    umap = None
    logger = logging.getLogger(__name__)
    logger.warning(
        "UMAP library not found. UMAP dimensionality reduction will be unavailable. "
        "Please install it with 'pip install umap-learn'."
    )

# Import FAISS, provide fallback warning if not installed
try:
    import faiss
    if TYPE_CHECKING:
        from faiss import Index as FAISSIndex
except ImportError:
    faiss = None
    logger = logging.getLogger(__name__)
    logger.warning(
        "FAISS library not found. Index loading functionality will be unavailable in fingerprint module. "
        "Please install it with 'pip install faiss-cpu' (or 'faiss-gpu')."
    )

logger = logging.getLogger(__name__)


def load_embeddings_from_index(
    index_path: str,
    sample_size: Optional[int] = None
) -> np.ndarray:
    """
    Loads embeddings from a FAISS index. If sample_size is provided,
    it reconstructs a random sample of vectors. Otherwise, it attempts
    to reconstruct all vectors (which might be memory intensive).

    Args:
        index_path: Path to the FAISS index file.
        sample_size: Number of vectors to sample from the index. If None,
                     attempts to load all vectors.

    Returns:
        A NumPy array of the loaded embeddings (N, D).

    Raises:
        FileNotFoundError: If the index file does not exist.
        RuntimeError: If FAISS is not installed or index cannot be loaded/reconstructed.
        ValueError: If index type does not support reconstruction and sample_size is not None.
    """
    if faiss is None:
        raise RuntimeError("FAISS is not installed. Cannot load embeddings for fingerprinting.")

    index_file_path = Path(index_path)
    if not index_file_path.exists():
        raise FileNotFoundError(f"FAISS index file not found: '{index_path}'.")

    logger.info(f"Loading FAISS index from '{index_path}' for fingerprinting...")
    try:
        index = faiss.read_index(str(index_file_path))
        logger.info(f"Loaded FAISS index: {type(index).__name__}, {index.ntotal} vectors, {index.d} dimensions.")
    except Exception as e:
        logger.error(f"Failed to load FAISS index '{index_path}': {e}")
        raise RuntimeError(f"Could not load FAISS index: {e}")

    if not index.is_trained:
        raise RuntimeError(f"FAISS index '{index_path}' is not trained. Cannot reconstruct vectors.")

    if not index.sa_code_size() > 0 and not isinstance(index, (faiss.IndexFlatL2, faiss.IndexFlatIP)):
        logger.warning(
            f"FAISS index type {type(index).__name__} might not support direct reconstruction "
            "without product quantization (PQ) or other compression. Attempting reconstruction."
        )
        # For non-flat indexes, direct reconstruction might not be ideal or supported without PQ.
        # Fallback to a simpler sampling if reconstruction is problematic.
        if sample_size is not None and sample_size > index.ntotal:
            sample_size = index.ntotal
        
        if sample_size is not None:
             # If index doesn't support reconstruction directly, or for large indexes,
             # a different strategy might be needed. For now, try to reconstruct.
             # In a production-ready tool, this would involve more sophisticated handling
             # of different FAISS index types or requiring raw vector files.
             logger.warning("Sampling from non-flat index by reconstructing subset. This might not represent full distribution accurately if index is heavily compressed.")
             
    if index.ntotal == 0:
        logger.warning(f"FAISS index '{index_path}' contains no vectors. Returning empty array.")
        return np.array([], dtype=np.float32).reshape(0, index.d)

    if sample_size is None or sample_size >= index.ntotal:
        # Reconstruct all vectors if no sample size specified or sample size is >= total
        logger.info(f"Reconstructing all {index.ntotal} vectors from index...")
        # Corrected: call reconstruct_n on the index object, not the faiss module
        vectors = index.reconstruct_n(0, index.ntotal)
    else:
        # Reconstruct a random sample of vectors
        logger.info(f"Reconstructing a random sample of {sample_size} vectors from index...")
        rng = np.random.default_rng()
        indices_to_sample = rng.choice(index.ntotal, size=sample_size, replace=False)
        # Corrected: call reconstruct_n on the index object, not the faiss module
        vectors = index.reconstruct_n(indices_to_sample[0], len(indices_to_sample))
        # Ensure we reconstruct exactly the sampled indices if reconstruct_n doesn't handle non-contiguous
        # Corrected: call reconstruct_n on the index object, not the faiss module
        if not np.array_equal(index.reconstruct_n(indices_to_sample[0], 1), index.reconstruct(indices_to_sample[0])): # rudimentary check
             vectors = np.array([index.reconstruct(int(i)) for i in indices_to_sample], dtype=np.float32)

    if vectors.ndim == 1: # Handle case where reconstruct_n returns 1D array for single vector
        vectors = vectors.reshape(1, -1)

    logger.info(f"Successfully loaded {vectors.shape[0]} embeddings with dimension {vectors.shape[1]}.")
    return vectors


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "umap",
    n_components: int = 2
) -> np.ndarray:
    """
    Reduces the dimensionality of embeddings using PCA or UMAP.

    Args:
        embeddings: A NumPy array of embeddings (N, D).
        method: The dimensionality reduction method ('pca' or 'umap').
        n_components: The target number of dimensions (e.g., 2 for 2D plot).

    Returns:
        A NumPy array of the reduced embeddings.

    Raises:
        ValueError: If an unsupported method is specified or UMAP is not installed.
    """
    if not embeddings.shape[0]:
        logger.warning("No embeddings to reduce. Returning empty array.")
        return np.array([])
    
    num_samples = embeddings.shape[0]

    # Decide on the effective reduction method
    effective_method = method.lower()
    
    # Force PCA for very small datasets where UMAP can be unstable or fail
    # UMAP generally needs more than just a few points (e.g., typically > 4-5) to build a meaningful manifold.
    # The spectral embedding can fail for very sparse or small graphs.
    if effective_method == "umap" and num_samples < 5: # Threshold of 5 is a common practical minimum for UMAP
        logger.warning(
            f"Number of samples ({num_samples}) is too small for meaningful UMAP projection. "
            "Forcing PCA for dimensionality reduction. Consider increasing --sample-size if possible."
        )
        effective_method = "pca" # Override method to PCA
    
    if embeddings.shape[1] < n_components:
        logger.warning(f"Number of components ({n_components}) is greater than embedding dimension ({embeddings.shape[1]}). Using original dimension.")
        return embeddings # Cannot reduce to more components than original dimensions


    logger.info(f"Reducing {num_samples} embeddings to {n_components} dimensions using {effective_method.upper()}...")
    if effective_method == "pca":
        reducer = PCA(n_components=n_components)
    elif effective_method == "umap":
        if umap is None:
            raise ValueError(
                "UMAP library not found. Cannot use 'umap' method. "
                "Please install it with 'pip install umap-learn'."
            )
        
        # Determine n_neighbors dynamically for UMAP to avoid issues with small datasets
        # n_neighbors must be less than the number of samples (embeddings.shape[0])
        # It also must be at least 2.
        # This part of the logic is now only reached if effective_method is UMAP and num_samples >= 5.
        adjusted_n_neighbors = min(15, max(2, num_samples - 1)) # Default 15, but cap at num_samples - 1, min 2

        logger.info(f"UMAP n_neighbors set to {adjusted_n_neighbors} for {num_samples} samples.")
        reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=adjusted_n_neighbors, min_dist=0.1)
    else:
        # This should ideally not be reached if effective_method is always 'pca' or 'umap'
        raise ValueError(f"Unsupported effective dimensionality reduction method: {effective_method}. Choose 'pca' or 'umap'.")

    try:
        reduced_embeddings = reducer.fit_transform(embeddings)
        logger.info(f"Successfully reduced embeddings to shape {reduced_embeddings.shape}.")
        return reduced_embeddings
    except Exception as e:
        logger.error(f"Error during dimensionality reduction with {effective_method}: {e}")
        raise RuntimeError(f"Dimensionality reduction failed: {e}")


def visualize_embeddings(
    reduced_embeddings_list: List[Tuple[np.ndarray, str]],
    output_path: str,
    title: str = "Embedding Distribution Fingerprint",
    xlabel: str = "Component 1",
    ylabel: str = "Component 2"
) -> None:
    """
    Generates and saves a scatter plot of reduced embeddings.

    Args:
        reduced_embeddings_list: A list of tuples, where each tuple contains
                                 (reduced_embeddings_array, label_string).
                                 Each array should be 2D (N, 2) or (N, 3).
        output_path: Path to save the plot (e.g., 'fingerprint.png', 'fingerprint.svg').
        title: Title of the plot.
        xlabel: Label for the X-axis.
        ylabel: Label for the Y-axis.

    Raises:
        ValueError: If reduced embeddings are not 2D.
        RuntimeError: If plotting fails.
    """
    if not reduced_embeddings_list:
        logger.warning("No reduced embeddings provided for visualization. Skipping plot generation.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    for embeddings, label in reduced_embeddings_list:
        if embeddings.ndim != 2 or embeddings.shape[1] < 2:
            logger.error(f"Reduced embeddings for label '{label}' are not 2D. Skipping this set.")
            continue
        ax.scatter(embeddings[:, 0], embeddings[:, 1], label=label, alpha=0.7, s=10)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    try:
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"Embedding fingerprint plot saved to: '{output_path}'.")
    except Exception as e:
        logger.error(f"Failed to save embedding fingerprint plot to '{output_path}': {e}")
        raise RuntimeError(f"Plotting failed: {e}")
    finally:
        plt.close(fig) # Close the figure to free up memory
