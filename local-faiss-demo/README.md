# Local FAISS RAG Demo

This folder contains a prototype implementation of a Retrieval-Augmented Generation (RAG) pipeline using FAISS for vector search and Amazon Bedrock for embedding and answering questions.

## Files

- **build_faiss_index.py**:
  - Downloads `.txt` files from a specified S3 prefix
  - Chunks the documents
  - Embeds each chunk with Titan Embeddings via Bedrock
  - Stores the vectors in a FAISS index and metadata in `metadata.json`

- **query_rag.py**:
  - Loads FAISS index + metadata
  - Accepts interactive user questions
  - Embeds question using Titan
  - Retrieves relevant chunks using FAISS
  - Constructs and submits a prompt to a Bedrock-hosted model (e.g., Claude)

## Usage

Build the index:
```bash
python build_faiss_index.py --bucket triviaqa \
                            --prefix evidence/wikipedia/ \
                            --region us-east-1 \
                            --index-path faiss.index \
                            --metadata-path metadata.json

python query_rag.py --index-path faiss.index \
                    --metadata-path metadata.json \
                    --region us-east-1 \
                    --model-id anthropic.claude-v2 \
                    --enable-rag
