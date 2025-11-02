# Cohere RAG Model — Internal Enablement POC

This repository provides a **Proof of Concept (POC)** implementation of a **Retrieval-Augmented Generation (RAG)** system that enables internal teams to **experiment with document and image search workflows**. The solution combines:

- **Cohere embeddings** for text similarity and PDF page retrieval  
- **FAISS** for lightweight vector search and local index persistence  
- **OpenAI Vision Models** for image-based question answering  
- **Streamlit** for a simple, interactive UI  

This POC is designed for **internal enablement**, rapid iteration, and dataset feasibility testing—not for production deployment.

## Purpose

This POC helps internal teams:

- Load and index PDFs with minimal friction
- Evaluate whether **RAG retrieval quality** meets expectations for the dataset
- Explore image-based Q&A and visual content search
- Validate early workflows before scaling toward enterprise deployment

## High-Level Architecture

User Query
   │
   │ (Text Embedding via Cohere)
   ▼
Vector Search (FAISS)  ← Indexed PDF Pages + Images
   │
   │ (Top-K Retrieved Context)
   ▼
LLM Response (OpenAI)
   │
   ▼
Streamlit UI Output

## Project Structure

Cohere_RAG_model-main/
│── app.py                         # Streamlit app entry point
│── config.py                      # API configuration & global paths
│── embeddings.py                  # Cohere text embedding utilities
│── faiss_utils.py                 # FAISS index persistence & vector operations
│── pdf_processing_embedding.py    # PDF → text → page-level embeddings
│── vision_query.py                # Image embedding + visual question answering
│── utils.py                       # File hashing, JSON helpers, support functions
│
├── source_docs/                   # PDFs to be processed and indexed
├── images/                        # Extracted images & uploaded images for search
├── store/                         # FAISS index + filename mappings (local persistence)
├── hashes/                        # File hash cache to prevent re-embedding duplicates
├── chat_data/                     # Optional conversation persistence
└── .env                           # API keys (not committed)

## Setup & Installation

1. Install dependencies:
pip install -r requirements.txt

2. Create a .env file:
COHERE_API_KEY=your_cohere_key
OPENAI_API_KEY=your_openai_key

3. Add PDFs to: /source_docs/

4. Run the app:
streamlit run app.py

## POC Constraints

- Single-machine, local index
- No RBAC or auth
- Not optimized for scaling
- No monitoring or observability

## Intended Audience

- Internal engineering + prototyping teams

## License

Internal use only.
