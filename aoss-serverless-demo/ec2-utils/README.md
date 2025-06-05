# EC2 Utilities for RAG with Bedrock + OpenSearch Serverless

This folder contains Python utilities for working with Retrieval-Augmented Generation (RAG) pipelines deployed on AWS infrastructure using OpenSearch Serverless (AOSS) and Amazon Bedrock.

## Utilities

### 1. `create_aoss_index.py`

Creates a k-NN enabled index in your OpenSearch Serverless collection for storing vector embeddings and metadata.

#### Usage:

```bash
python create_aoss_index.py \
  --region us-east-1 \
  --aoss-endpoint https://<your-aoss-endpoint> \
  --index triviaqa
```

---

### 2. `embed_to_aoss.py`

Processes text files from S3: chunking, embedding via Titan, and indexing into AOSS.

#### Usage:

```bash
python embed_to_aoss.py \
  --region us-east-1 \
  --bucket triviaqa \
  --prefix evidence/wikipedia/ \
  --aoss-endpoint https://<your-aoss-endpoint> \
  --index triviaqa \
  --embed-model-id amazon.titan-embed-text-v1
```

---

### 3. `query_rag.py`

Runs a command-line RAG loop using Titan for embedding and generation, with optional RAG.

#### Usage:

```bash
python query_rag.py \
  --region us-east-1 \
  --index triviaqa \
  --aoss-endpoint https://<your-aoss-endpoint> \
  --embed-model-id amazon.titan-embed-text-v1 \
  --gen-model-id amazon.titan-text-lite-v1 \
  --top-k 5 \
  --enable-rag
```

To disable RAG (use generation only):

```bash
python query_rag.py --region us-east-1 --index triviaqa --aoss-endpoint <...>
```

