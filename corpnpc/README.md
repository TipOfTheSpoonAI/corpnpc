CORPNPC: Compliance Oversight & Risk Probe for Neural Parrot Convergence
Operational Imperative
The proliferation of Large Language Models (LLMs) introduces novel vectors for data exfiltration, intellectual property compromise, and provenance obfuscation. CORPNPC is a forensic toolkit engineered to neutralize these threats by surfacing hidden convergence within LLM embedding spaces.

Mission Statement
CORPNPC empowers U.S.-aligned cyber defense operations—collectively referred to as Patriot Prime—to detect and attribute adversarial exploitation of LLM technologies. It enables forensic identification of unauthorized data convergence within foreign-sourced models, especially those influenced by the Communist Cyber Pirates (CCP).

Key Capabilities
CORPNPC provides three primary analytical modules, each addressing a distinct adversarial vector:

1. crack (Embedding Leak Detection)
Function: Probes target FAISS indexes (representing LLM embedding spaces) with specified n-gram corpora to identify statistically significant similarity matches.

CCP Context: Confirms inclusion of memorized U.S. IP, defense specifications, or industrial designs.

Patriot Prime Use Case: Forensically probe adversarial LLMs for indicators of data theft.

2. fingerprint (Embedding Distribution Analysis)
Function: Analyzes and visualizes the geometric distribution of embeddings from one or more FAISS indexes. Useful for detecting "shadow clones" or models with highly similar underlying structures.

CCP Context: Exposes mimicry or covert lineage reuse even across rebranded models.

Patriot Prime Use Case: Validate if an LLM shares latent structure with known U.S. models.

3. entropy-scan (Rare Phrase Detection)
Function: Identifies phrases that appear in a sensitive corpus AND are statistically rare (high entropy) in a larger background corpus. Useful for detecting leaked proprietary terms.

CCP Context: Reveals inclusion of low-frequency classified jargon or operational security breach indicators.

Patriot Prime Use Case: Confirm compromise of U.S. operational language at the phrase level.

Key Features
Architecture & Performance
Pluggable embedding backends (BaseEmbedder interface)

FAISS integration for fast nearest-neighbor search

Concurrent probing support

Deterministic test mode

Configurable cosine similarity threshold

Detection & Intelligence
Bidirectional leakage detection

Rare phrase entropy scanning

Embedding distribution fingerprinting

Transparency & Output
Granular logging with performance metrics

JSON leak reports with match, score, confidence

Easy CLI usage with full module options

Adversary Profile: Communist Cyber Pirates (CCP)
A persistent state-aligned threat actor focused on AI-enabled espionage, technological mimicry, and covert convergence of stolen datasets.

Beneficiary Profile: U.S. Patriot Prime
Any authorized American entity tasked with defending national cyber infrastructure—including military, intelligence, and federal cyber command units.

Installation
Prerequisites
Python 3.8+

pip (Python package installer)

Setup Steps
Clone the Repository:

Linux/macOS:

git clone https://github.com/TipOfTheSpoonAI/corpnpc.git
cd corpnpc

PowerShell (Windows):

git clone https://github.com/TipOfTheSpoonAI/corpnpc.git
cd corpnpc

Create a Virtual Environment (Highly Recommended):

Linux/macOS:

python -m venv venv
source venv/bin/activate

PowerShell (Windows):

python -m venv venv
.\venv\Scripts\activate

Install Dependencies:
CORPNPC relies on several powerful libraries.

pip install typer numpy scikit-learn matplotlib umap-learn Faker
pip install "sentence-transformers>=2.2.0"
pip install "faiss-cpu>=1.7.0"             # For CPU-only FAISS. For NVIDIA GPUs, use: pip install "faiss-gpu>=1.7.0"

Note: If faiss-gpu is installed, ensure CUDA toolkit and GPU drivers are correctly configured and compatible with the FAISS build. Refer to the FAISS GitHub page for detailed GPU installation instructions.

Usage
The primary entry point for CORPNPC is the cli.py script.

Usage: python -m corpnpc.cli [OPTIONS] COMMAND [ARGS]...

CORPNPC: Compliance Oversight & Risk Probe for Neural Parrot Convergence. A forensic toolkit for detecting and analyzing
data leakage in LLM embedding spaces.


╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                            │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ crack           Cracks a FAISS index to detect potential embedding leaks by probing it with a given n-gram corpus.     │
│ fingerprint     Analyzes and visualizes the geometric distribution of embeddings from one or more FAISS indexes. Useful│
│                 for detecting "shadow clones" or models with highly similar underlying structures.                     │
│ entropy-scan    Identifies phrases that appear in a sensitive corpus AND are statistically rare (high entropy) in a    │
│                 larger background corpus. Useful for detecting leaked proprietary terms.                               │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

1. crack Command (Embedding Leak Detection)
To perform a basic leak detection scan, the following inputs are required:

A pre-built FAISS index file (e.g., vectors.faiss).

A plaintext .txt file containing the n-gram corpus for probing (one n-gram/phrase per line).

Usage: python -m corpnpc.cli crack [OPTIONS]

Cracks a FAISS index to detect potential embedding leaks by probing it with a given n-gram corpus.

The process involves embedding the n-grams, querying the FAISS index for similar vectors, filtering for high-confidence
unique matches, and generating a detailed leak report.

╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                            │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Crack Options ────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ * --index  -i      FILE          Path to the pre-built FAISS index file (e.g.,                                     │
│                                   './vectors.faiss').                                                                 │
│                                   [default: None]                                                                      │
│                                   [required]                                                                           │
│ * --ngrams -n      FILE          Path to the plaintext .txt file containing the n-gram corpus                       │
│                                   (one n-gram per line).                                                               │
│                                   [default: None]                                                                      │
│                                   [required]                                                                           │
│    --threshold -t   FLOAT RANGE [0.0<=x<=1.0]  Cosine similarity threshold (0.0 to 1.0). Matches with scores         │
│                                                below this value will be filtered out. Higher values mean             │
│                                                higher confidence leaks.                                              │
│                                                [default: 0.9]                                                          │
│    --bidirectional                             Enable bidirectional probing (A->B and B->A). Requires a              │
│                                                second FAISS index and n-gram corpus for the reverse                  │
│                                                direction.                                                              │
│    --reverse-index -ri  FILE          Path to the FAISS index for the reverse probing direction                      │
│                                   (B->A). Required with --bidirectional.                                               │
│                                   [default: None]                                                                      │
│    --reverse-ngrams -rn  FILE          Path to the n-gram corpus for the reverse probing direction                   │
│                                   (B->A). Required with --bidirectional.                                               │
│                                   [default: None]                                                                      │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Embedding Options ────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --backend -b      TEXT          Embedding backend to use. Options: 'sentence-transformer'. [default: sentence-transformer] │
│ --model -m        TEXT          Name of the specific model to use for the chosen embedding backend (e.g., 'all-MiniLM-L6-v2' │
│                                 for Sentence-Transformer).                                                               │
│                                 [default: all-MiniLM-L6-v2]                                                              │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Reporting Options ────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --output -o       PATH          Path to save the leak detection report in JSON format.                                 │
│                                 [default: corpnpc_crack_leak_report.json]                                              │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Development/Testing ──────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --test-mode                   Inject a known duplicate vector to validate matching, bypassing normal embedding. Requires the │
│                               FAISS index to be accessible for fetching a known vector.                                │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Performance Options ──────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --concurrent                  Enable concurrent execution for bidirectional probing. Reduces total runtime for large datasets. │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Logging Options ──────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --verbose -v                  Enable verbose output (INFO level logging).                                              │
│ --debug                       Enable debug output (DEBUG level logging). Overrides --verbose.                          │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Examples for crack:

Basic Scan with Custom Output:

Linux/macOS:

python -m corpnpc.cli crack \
    -i ./my_data/product_embeddings.faiss \
    -n ./my_data/proprietary_phrases.txt \
    -o ./reports/product_leak_summary.json \
    -t 0.92 --verbose

PowerShell (Windows):

python -m corpnpc.cli crack `
    -i ./my_data/product_embeddings.faiss `
    -n ./my_data/proprietary_phrases.txt `
    -o ./reports/product_leak_summary.json `
    -t 0.92 --verbose

Running with Bidirectional Probing (A->B and B->A):
This is crucial for identifying asymmetric knowledge transfer.

Linux/macOS:

python -m corpnpc.cli crack \
    -i ./model_A_index.faiss -n ./model_B_ngrams.txt \
    --bidirectional \
    -ri ./model_B_index.faiss -rn ./model_A_ngrams.txt \
    -o ./reports/bidirectional_leak_analysis.json \
    -t 0.95 --verbose --debug

PowerShell (Windows):

python -m corpnpc.cli crack `
    -i ./model_A_index.faiss -n ./model_B_ngrams.txt `
    --bidirectional `
    -ri ./model_B_index.faiss -rn ./model_A_ngrams.txt `
    -o ./reports/bidirectional_leak_analysis.json `
    -t 0.95 --verbose --debug

model_B_ngrams.txt will be probed against model_A_index.faiss.

model_A_ngrams.txt will be probed against model_B_index.faiss.

Using the Test Mode (for Development/Debugging):
This will inject a perfect match from vectors.faiss to ensure the core detection logic works.

Linux/macOS:

python -m corpnpc.cli crack \
    --index vectors.faiss --ngrams ngrams.txt \
    --output test_report.json --threshold 0.90 \
    --test-mode --verbose

PowerShell (Windows):

python -m corpnpc.cli crack `
    --index vectors.faiss --ngrams ngrams.txt `
    --output test_report.json --threshold 0.90 `
    --test-mode --verbose

2. fingerprint Command (Embedding Distribution Analysis)
Usage: python -m corpnpc.cli fingerprint [OPTIONS]

Analyzes and visualizes the geometric distribution of embeddings from one or more FAISS indexes. Useful for detecting
"shadow clones" or models with highly similar underlying structures.


╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                            │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Fingerprint Options ──────────────────────────────────────────────────────────────────────────────────────────────────╮
│ * --index  -i      FILE          Path(s) to the FAISS index files to fingerprint. Can specify                     │
│                                   multiple times.                                                                    │
│                                   [default: None]                                                                      │
│                                   [required]                                                                           │
│    --sample-size -s INTEGER RANGE [x>=0]   Number of vectors to sample from each index for dimensionality            │
│                                            reduction and plotting. Set to 0 or None to attempt to load all           │
│                                            (memory intensive for large indexes).                                     │
│                                            [default: 10000]                                                          │
│    -r             TEXT          Dimensionality reduction method: 'umap' (default, requires                          │
│                                 umap-learn) or 'pca'.                                                                │
│                                 [default: umap]                                                                      │
│    --n-components -c INTEGER RANGE [2<=x<=3]   Number of dimensions to reduce to (typically 2 or 3 for              │
│                                                visualization).                                                       │
│                                                [default: 2]                                                            │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Reporting Options ────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --output -o       PATH          Path to save the embedding fingerprint plot (e.g., '.png', '.svg').                  │
│                                 [default: corpnpc_fingerprint_plot.png]                                              │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Logging Options ──────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --verbose -v                  Enable verbose output (INFO level logging).                                              │
│ --debug                       Enable debug output (DEBUG level logging). Overrides --verbose.                          │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Examples for fingerprint:

Compare two FAISS indexes using UMAP:

Linux/macOS:

python -m corpnpc.cli fingerprint \
    --index ./testbed_small/index_A.faiss \
    --index ./testbed_small/index_B.faiss \
    --output ./reports/embedding_comparison.png \
    --sample-size 5000 --verbose

PowerShell (Windows):

python -m corpnpc.cli fingerprint `
    --index ./testbed_small/index_A.faiss `
    --index ./testbed_small/index_B.faiss `
    --output ./reports/embedding_comparison.png `
    --sample-size 5000 --verbose

Fingerprint a single index using PCA to 3D:

Linux/macOS:

python -m corpnpc.cli fingerprint \
    -i ./my_model_index.faiss \
    -o ./reports/model_fingerprint_3d.png \
    -s 10000 \
    -r pca \
    -c 3

PowerShell (Windows):

python -m corpnpc.cli fingerprint `
    -i ./my_model_index.faiss `
    -o ./reports/model_fingerprint_3d.png `
    -s 10000 `
    -r pca `
    -c 3

3. entropy-scan Command (Rare Phrase Detection)
Usage: python -m corpnpc.cli entropy-scan [OPTIONS]

Identifies phrases that appear in a sensitive corpus AND are statistically rare (high entropy) in a larger background
corpus. Useful for detecting leaked proprietary terms.


╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                            │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Entropy Scan Options ─────────────────────────────────────────────────────────────────────────────────────────────────╮
│ * --sensitive-ngrams -s FILE          Path to the plaintext .txt file containing the sensitive                     │
│                                        n-gram corpus (one n-gram per line).                                          │
│                                        [default: None]                                                               │
│                                        [required]                                                                    │
│ * --background-corpus -b FILE          Path to a larger, general plaintext .txt file                                │
│                                        representing the background data (e.g., a dump of LLM                         │
│                                        training data or public corpus).                                              │
│                                        [default: None]                                                               │
│                                        [required]                                                                    │
│    --ngram-size -n INTEGER RANGE [x>=1]    The size of n-grams to consider (e.g., 1 for unigrams, 2                 │
│                                            for bigrams).                                                             │
│                                            [default: 1]                                                              │
│    --rarity-percentile -r FLOAT RANGE [0.0<=x<=100.0]    A percentile (0-100). Only n-grams with a frequency below │
│                                                        this percentile in the background corpus will be            │
│                                                        considered 'rare' and potentially leaked. Lower values      │
│                                                        mean rarer phrases.                                         │
│                                                        [default: 20.0]                                               │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Reporting Options ────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --output -o       PATH          Path to save the entropy scan report in JSON format. [default: corpnpc_entropy_report.json] │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Logging Options ──────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --verbose -v                  Enable verbose output (INFO level logging).                                              │
│ --debug                       Enable debug output (DEBUG level logging). Overrides --verbose.                          │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Examples for entropy-scan:

Scan for rare unigrams in sensitive data:

Linux/macOS:

python -m corpnpc.cli entropy-scan \
    -s ./testbed_small/sensitive.txt \
    -b ./testbed_small/background.txt \
    -o ./reports/rare_unigrams_report.json \
    -n 1 --rarity-percentile 10.0 --verbose

PowerShell (Windows):

python -m corpnpc.cli entropy-scan `
    -s ./testbed_small/sensitive.txt `
    -b ./testbed_small/background.txt `
    -o ./reports/rare_unigrams_report.json `
    -n 1 --rarity-percentile 10.0 --verbose

Scan for rare bigrams with a higher percentile threshold:

Linux/macOS:

python -m corpnpc.cli entropy-scan \
    --sensitive-ngrams ./classified_terms.txt \
    --background-corpus ./public_web_crawl.txt \
    --output ./reports/classified_bigrams.json \
    --ngram-size 2 --rarity-percentile 5.0

PowerShell (Windows):

python -m corpnpc.cli entropy-scan `
    --sensitive-ngrams ./classified_terms.txt `
    --background-corpus ./public_web_crawl.txt `
    --output ./reports/classified_bigrams.json `
    --ngram-size 2 --rarity-percentile 5.0

Project Structure
cli.py: The main command-line interface, responsible for parsing arguments, orchestrating the overall leak detection workflow, and initiating probes.

corpnpc/: The core package directory.

__init__.py: Initializes the package.

embed.py: Manages embedding generation. It defines the BaseEmbedder abstract interface, provides the SentenceTransformerEmbedder concrete implementation, and includes the get_embedder factory function for selecting embedding backends. It also contains the embed_corpus function (with test_mode integration).

crack.py: Contains functions for loading FAISS indexes and performing similarity probing against them.

score.py: Implements logic for filtering raw similarity matches based on thresholds and uniqueness, and assigns confidence labels.

report.py: Handles the generation and saving of the final leak detection report in JSON (or JSON Lines) format.

fingerprint.py: Contains functions for loading embeddings, performing dimensionality reduction (PCA/UMAP), and visualizing embedding distributions.

entropy.py: Implements logic for analyzing text corpora to identify rare and common phrases for leakage detection.

Extensibility: Adding New Embedding Backends
The modular design of CORPNPC makes it straightforward to integrate new embedding providers (e.g., commercial APIs like OpenAI, Cohere, or other local models).

To add a new backend:

Create a New Embedder Class: In corpnpc/embed.py, create a new Python class that inherits from BaseEmbedder.

# Example for a hypothetical OpenAIEmbedder
from corpnpc.embed import BaseEmbedder
import numpy as np
from typing import List

class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, api_key: str):
        # Initialize your OpenAI client here
        self.client = OpenAI(api_key=api_key) # Assuming 'OpenAI' client

    def embed(self, texts: List[str]) -> np.ndarray:
        # Implement the embedding logic using the OpenAI API
        response = self.client.embeddings.create(
            model="text-embedding-ada-002", # Or whatever model name
            input=texts
        )
        embeddings = [d.embedding for d in response.data]
        return np.array(embeddings, dtype=np.float32)

Register in get_embedder Factory: In the get_embedder function within corpnpc/embed.py, add an elif condition to return an instance of your new embedder when its corresponding backend name is requested.

def get_embedder(backend: str, model_name: str) -> BaseEmbedder:
    backend = backend.lower()
    if backend == "sentence-transformer":
        return SentenceTransformerEmbedder(model_name)
    elif backend == "openai": # New entry
        # You might need to pass API keys or other credentials here.
        # This would involve adding new CLI options to `cli.py` to capture them.
        return OpenAIEmbedder(api_key="YOUR_OPENAI_API_KEY")
    # ... other backends
    else:
        raise ValueError(f"Unknown embedding backend: '{backend}'. "
                         "Supported options: 'sentence-transformer'.")

Update cli.py (if necessary): If your new backend requires specific API keys or parameters, you'll need to add new typer.Option arguments to the crack command in cli.py to capture these from the user and pass them to your new embedder's __init__ method via get_embedder.

Contributing
Not required. Not expected. Fork it or don’t.

License
This project is licensed under the MIT License.

Acknowledgements
Authored by: @TipOfTheSpoonAI on X

Extending gratitude to the creators and maintainers of the following open-source projects, whose foundational work makes CORPNPC possible:

FAISS (Facebook AI Similarity Search): Developed by Facebook AI.

Sentence-Transformers: A project built on top of Hugging Face Transformers. The all-MiniLM-L6-v2 model, used as a default, is a highly efficient and effective model for sentence embeddings.

Typer: For providing an intuitive and powerful framework for building command-line applications.

NumPy: The fundamental package for numerical computing with Python.

scikit-learn: For robust machine learning utilities, including vector normalization.

UMAP (Uniform Manifold Approximation and Projection): For non-linear dimensionality reduction and visualization.

Matplotlib: For creating static, interactive, and animated visualizations in Python.

Faker: For generating realistic fake data for testing purposes.