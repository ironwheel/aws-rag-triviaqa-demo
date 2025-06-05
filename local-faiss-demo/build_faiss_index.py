#!/usr/bin/env python3
"""
@file build_faiss_index.py
@brief Build FAISS vector index from S3 text files using Amazon Titan embeddings (via Bedrock).
@details
    - Lists and downloads text files from an S3 bucket
    - Splits them into word chunks
    - Embeds each chunk using Bedrock Titan
    - Stores vectors into FAISS
    - Outputs metadata for chunk traceability

@copyright Robert E. Taylor, Extropic Systems, 2025
@license MIT
"""

import boto3
import json
import faiss
import numpy as np
import argparse
from pathlib import Path
from typing import List
from botocore.exceptions import ClientError

def list_text_files(bucket: str, prefix: str, max_files: int) -> List[str]:
    """
    @brief List up to `max_files` text files under a given S3 prefix.
    @param bucket S3 bucket name
    @param prefix Prefix path under the bucket
    @param max_files Max number of files to retrieve
    @return List of object keys for .txt files
    """
    paginator = s3.get_paginator('list_objects_v2')
    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.txt'):
                files.append(obj['Key'])
                if len(files) >= max_files:
                    return files
    return files

def get_s3_text(bucket: str, key: str) -> str:
    """
    @brief Download text file contents from S3.
    @param bucket S3 bucket name
    @param key Full S3 key (object path)
    @return UTF-8 decoded string contents
    """
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj['Body'].read().decode('utf-8')

def chunk_text(text: str, max_words: int = 200, overlap: int = 50) -> List[str]:
    """
    @brief Split a large string into word-based chunks with overlap.
    @param text Input full document text
    @param max_words Maximum words per chunk
    @param overlap Number of overlapping words between chunks
    @return List of chunked text segments
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunk = ' '.join(words[i:i + max_words])
        if chunk:
            chunks.append(chunk)
    return chunks

def get_titan_embedding(text_chunk: str) -> List[float]:
    """
    @brief Generate a vector embedding using Titan via Bedrock.
    @param text_chunk Text input to embed
    @return List of floats representing the embedding
    """
    body = json.dumps({"inputText": text_chunk})
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    response_body = json.loads(response['body'].read())
    return response_body['embedding']

def process_file(bucket: str, key: str, index, metadata_list):
    """
    @brief Chunk and embed a single S3 text file and add results to FAISS and metadata list.
    @param bucket S3 bucket name
    @param key S3 key of the text file
    @param index FAISS index object
    @param metadata_list List to append metadata for each chunk
    """
    print(f"\nProcessing: {key}")
    text = get_s3_text(bucket, key)
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)}")
        embedding = get_titan_embedding(chunk)
        vector = np.array(embedding, dtype='float32')
        index.add(np.expand_dims(vector, axis=0))
        metadata_list.append({
            "chunk": chunk,
            "source": key
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from S3-hosted text files using Titan Embeddings.")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", default="evidence/wikipedia/", help="S3 prefix path (default: evidence/wikipedia/)")
    parser.add_argument("--max-files", type=int, default=10, help="Maximum number of files to process")
    parser.add_argument("--index-path", default="faiss.index", help="Output FAISS index file path")
    parser.add_argument("--metadata-path", default="metadata.json", help="Output metadata JSON file path")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")

    args = parser.parse_args()

    # AWS clients
    bedrock = boto3.client('bedrock-runtime', region_name=args.region)
    s3 = boto3.client('s3')

    # FAISS index and metadata store
    embedding_dim = 1536
    index = faiss.IndexFlatL2(embedding_dim)
    metadata_list = []

    print("Listing files in S3...")
    files = list_text_files(args.bucket, args.prefix, args.max_files)
    print(f"Found {len(files)} files. Starting processing...")

    for key in files:
        process_file(args.bucket, key, index, metadata_list)

    print(f"\nSaving FAISS index to: {args.index_path}")
    faiss.write_index(index, args.index_path)

    print(f"Saving metadata to: {args.metadata_path}")
    with open(args.metadata_path, "w") as f:
        json.dump(metadata_list, f, indent=2)

    print("Done.")

