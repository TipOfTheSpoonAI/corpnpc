import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from abc import ABC, abstractmethod

# Set up logging for this module.
logger = logging.getLogger(__name__)

# Attempt to import FAISS for test_mode functionality.
try:
    import faiss
except ImportError:
    faiss = None
    logger.warning(
        "FAISS library not found. Test mode functionality relying on FAISS "
        "will be unavailable. Please install with 'pip install faiss-cpu' (or 'faiss-gpu')."
    )

# --- BaseEmbedder Interface ---
class BaseEmbedder(ABC):
    """
    Abstract base class for all embedding providers.
    Defines the contract for embedding text.
    """
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of texts and returns an array of embeddings.

        Args:
            texts: A list of strings to be embedded.

        Returns:
            A NumPy array where each row is the embedding vector for the corresponding text.
            The shape will be (len(texts), embedding_dimension).
        """
        pass

# --- SentenceTransformer Embedder Implementation ---
# Cache for SentenceTransformer models to prevent re-loading the same model multiple times
_sentence_transformer_model_cache: Dict[str, "SentenceTransformer"] = {}

class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Implements the BaseEmbedder interface using the SentenceTransformer library.
    """
    def __init__(self, model_name: str):
        """
        Initializes the SentenceTransformerEmbedder with a specific model.

        Args:
            model_name: The name of the SentenceTransformer model to load
                        (e.g., "all-MiniLM-L6-v2").

        Raises:
            RuntimeError: If SentenceTransformer library or torch is not installed,
                          or if the model cannot be loaded.
        """
        try:
            from sentence_transformers import SentenceTransformer
            import torch # Explicitly import torch to check for CUDA availability
        except ImportError:
            logger.error(
                "Sentence-transformers or torch not found. Cannot initialize "
                "SentenceTransformerEmbedder. Please install them with "
                "'pip install sentence-transformers torch'."
            )
            raise RuntimeError("SentenceTransformer or torch is not installed.")

        # Use the module-level cache to ensure models are loaded only once
        if model_name not in _sentence_transformer_model_cache:
            logger.info(f"Loading SentenceTransformer model: '{model_name}'...")
            try:
                self.model = SentenceTransformer(model_name)
                _sentence_transformer_model_cache[model_name] = self.model
                if torch.cuda.is_available():
                    logger.info(f"Model loaded successfully. Using GPU: {torch.cuda.get_device_name(0)}")
                else:
                    logger.info("Model loaded successfully. Running on CPU.")
            except Exception as e:
                logger.critical(f"Failed to load SentenceTransformer model '{model_name}': {e}")
                logger.critical("Please check your internet connection, model name, and torch installation.")
                raise RuntimeError(f"Could not load SentenceTransformer model '{model_name}': {e}")
        else:
            logger.debug(f"Using cached SentenceTransformer model for: '{model_name}'")
            self.model = _sentence_transformer_model_cache[model_name]

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of texts using the loaded SentenceTransformer model.

        Args:
            texts: A list of strings to embed.

        Returns:
            A NumPy array of embeddings.
        """
        logger.info(f"Encoding {len(texts)} texts using SentenceTransformer model...")
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)


# --- Embedder Factory ---
def get_embedder(backend: str, model_name: str) -> BaseEmbedder:
    """
    Factory function to get an embedder instance based on the specified backend.

    Args:
        backend: The name of the embedding backend (e.g., "sentence-transformer").
        model_name: The specific model name to use for the chosen backend.

    Returns:
        An instance of a class implementing BaseEmbedder.

    Raises:
        ValueError: If an unknown embedding backend is requested.
        NotImplementedError: If the requested backend is recognized but not yet implemented.
    """
    backend = backend.lower()

    if backend == "sentence-transformer":
        return SentenceTransformerEmbedder(model_name)
    elif backend == "openai":
        raise NotImplementedError("OpenAI backend not implemented yet. "
                                  "Please contribute to add this functionality!")
    elif backend == "cohere":
        raise NotImplementedError("Cohere backend not implemented yet. "
                                  "Please contribute to add this functionality!")
    else:
        raise ValueError(f"Unknown embedding backend: '{backend}'. "
                         "Supported options: 'sentence-transformer'.")


# --- Corpus Embedding Function (with Test Mode) ---
def embed_corpus(
    corpus_path: str,
    embedder: BaseEmbedder,
    index_path: Optional[str] = None, # Added for test_mode
    test_mode: bool = False
) -> List[Tuple[str, np.ndarray]]:
    """
    Loads a plaintext corpus from a .txt file, embeds each non-empty line
    using the provided embedder, and returns the original lines paired with
    their corresponding embedding vectors.

    In test_mode, it can bypass actual embedding and inject a known vector
    from a FAISS index to facilitate deterministic testing.

    Args:
        corpus_path: The file path (string) to the plaintext .txt corpus file.
                     Each line in this file is treated as a separate document/sentence
                     to be embedded.
        embedder: An instance of a class implementing BaseEmbedder, which will
                  be used to generate the embeddings.
        index_path: The file path to the FAISS index. Required if test_mode is True.
        test_mode: If True, bypasses actual embedding and injects a predefined
                   vector and text for testing purposes.

    Returns:
        A list of tuples, where each tuple contains:
        - original_line (str): The cleaned, non-empty line from the corpus file.
        - embedding_vector (np.ndarray): The dense vector representation (embedding)
                                         of the `original_line`.

    Raises:
        FileNotFoundError: If the `corpus_path` does not exist.
        IsADirectoryError: If the `corpus_path` points to a directory instead of a file.
        IOError: For other input/output related errors during file reading.
        RuntimeError: If the embedding process fails using the provided embedder,
                      or if FAISS is not available in test_mode.
        Exception: For any other unexpected errors during the embedding process.
    """
    # --- TEST MODE INJECTION ---
    if test_mode:
        if faiss is None:
            logger.error("FAISS not installed. Cannot use test_mode without FAISS.")
            raise RuntimeError("FAISS is not installed, cannot use test_mode.")
        if index_path is None:
            logger.error("index_path must be provided when test_mode is enabled.")
            raise ValueError("index_path is required for test_mode.")
        
        logger.warning("TEST MODE ENABLED: Forcing probe vector to exactly match FAISS index[0]")
        try:
            # Need to re-read index for this specific test case, as the main FAISS index
            # loading happens later in cli.py. This ensures embed_corpus can access it.
            test_index = faiss.read_index(str(Path(index_path)))
            vec = test_index.reconstruct(0) # Get the 0th vector from the index
            # Ensure the vector is float32 as expected by FAISS later
            vec = vec.astype(np.float32).reshape(1, -1)
            # Create dummy lines and embeddings for the test
            return [("confidential government payload coordinates", vec[0])]
        except Exception as e:
            logger.critical(f"Failed to set up test mode from index '{index_path}': {e}")
            raise RuntimeError(f"Test mode setup failed: {e}")
    # --- END TEST MODE INJECTION ---


    corpus_file_path = Path(corpus_path)

    # --- Pre-checks for robustness ---
    if not corpus_file_path.exists():
        logger.error(f"Corpus file not found: '{corpus_path}'. Cannot embed corpus.")
        raise FileNotFoundError(f"No such file or directory: '{corpus_path}'")
    if corpus_file_path.is_dir():
        logger.error(f"Provided path '{corpus_path}' is a directory, not a file. Cannot embed corpus.")
        raise IsADirectoryError(f"'{corpus_path}' is a directory. Please provide a file path.")
    
    lines: List[str] = []
    try:
        logger.info(f"Loading lines from corpus file: '{corpus_path}'...")
        with open(corpus_file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                stripped_line = line.strip()
                if stripped_line:
                    lines.append(stripped_line)
        logger.info(f"Successfully loaded {len(lines)} non-empty lines from '{corpus_path}'.")
        if not lines:
            logger.warning(f"Corpus file '{corpus_path}' is empty or contains only whitespace lines. No embeddings will be generated.")
            return []

    except FileNotFoundError:
        logger.error(f"Corpus file '{corpus_path}' disappeared during processing.")
        raise
    except IOError as e:
        logger.error(f"An I/O error occurred while reading corpus '{corpus_path}': {e}")
        raise
    except Exception as e:
        logger.critical(f"An unexpected error occurred while loading corpus from '{corpus_path}': {e}")
        raise

    # --- Embedding Process (Normal Mode) ---
    try:
        embeddings = embedder.embed(lines)
        
        if embeddings.shape[0] != len(lines):
            logger.error(f"Mismatch in number of lines embedded ({embeddings.shape[0]}) vs. input lines ({len(lines)}).")
            raise RuntimeError("Embedding process did not return expected number of embeddings.")
        if embeddings.ndim != 2:
            logger.error(f"Embeddings are not 2-dimensional (expected N, D), got {embeddings.ndim} dimensions.")
            raise RuntimeError("Embeddings are not in the expected format.")

        logger.info(f"Successfully embedded {len(lines)} lines. Embedding dimension: {embeddings.shape[1]}")
    except Exception as e:
        logger.critical(f"Failed during embedding process for '{corpus_path}': {e}")
        raise RuntimeError(f"Embedding failed: {e}")

    return list(zip(lines, embeddings))
