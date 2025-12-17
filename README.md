**BACTERIA_PARSER**

A pipeline for extracting morphology evidence from microbiology textbooks and determining whether each bacterium is multicellular, unicellular, or unclear — using embeddings, hybrid indexing, and Ollama language models.

**1.  Embeddings Builder (embeddings_builder_HUG.py)**
Converts scientific books (.txt) into searchable embedding chunks.

What it does:

Splits each book into overlapping chunks

Generates embeddings using SentenceTransformer

Stores results as .json files in embeddings/

Why:
Allows fast semantic search for taxonomic names and morphology descriptions.

**2.  Hybrid Taxonomy Index (INDEX_FROM_EMBED.py)**

Searches the embedding dataset for all taxonomy levels (Order, Family, Genus, Species).

What it does:

Reads lists of taxon names from text files

Builds augmented search queries with morphology keywords

Performs semantic + keyword (“hybrid”) search

Saves results as *_hybrid_matches.json

Why:
Collects all relevant evidence from the books before querying the language model.

**3.  Ollama Evaluation (OLAMA_SEARCH.py)**

Uses an LLM (e.g., mistral:7b-instruct via Ollama) to evaluate extracted evidence.

Key features:

Multi-pass extraction of verbatim morphological sentences

Automatic cleanup of malformed OCR words

Final curator prompt decides: True / False / UNCLEAR

Writes one JSON report per species to ollama_results/

