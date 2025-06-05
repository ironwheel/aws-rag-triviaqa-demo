""#!/usr/bin/env python3
"""
@file embed_to_aoss.py
@brief Embeds text chunks from S3 using Bedrock Titan and stores them in an OpenSearch Serverless (AOSS) index.

@copyright Robert E. Taylor, Extropic Systems, 2025
@license MIT
"""

import boto3
import json
import requests
from requests_aws4auth import AWS4Auth
import re
import time
import argparse

# --- Command-line Argument Parsing ---
parser = argparse.ArgumentParser(description="Embed text chunks from S3 using Bedrock and store them in AOSS.")
parser.add_argument("--region", required=True, help="AWS region (e.g., us-east-1)")
parser.add_argument("--bucket", required=True, help="S3 bucket name")
parser.add_argument("--prefix", default="evidence/wikipedia/", help="Prefix in S3 bucket to scan for .txt files")
parser.add_argument("--index", required=True, help="OpenSearch index name")
parser.add_argument("--aoss-endpoint", required=True, help="OpenSearch Serverless endpoint URL")
parser.add_argument("--bedrock-model-id", default="amazon.titan-embed-text-v1", help="Bedrock embedding model ID")
args = parser.parse_args()

# --- AWS Session and Clients ---
session = boto3.Session(region_name=args.region)
credentials = session.get_credentials().get_frozen_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    args.region,
    "aoss",
    session_token=credentials.token,
)

s3 = session.client("s3")
bedrock = session.client("bedrock-runtime", region_name=args.region)

# --- Chunking Utility ---
def chunk_text(text, max_words=200):
    """
    @brief Split text into chunks of max_words.
    @param text Full document text
    @param max_words Maximum number of words per chunk
    @return List of text chunks
    """
    print("🔧 Chunking text...")
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    count = 0
    for s in sentences:
        w = len(s.split())
        if count + w > max_words:
            chunks.append(" ".join(current))
            current, count = [s], w
        else:
            current.append(s)
            count += w
    if current:
        chunks.append(" ".join(current))
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# --- Embedding Utility ---
def embed_text(text):
    """
    @brief Embed a text chunk using Titan
    @param text Text chunk to embed
    @return List of floats (embedding vector)
    """
    print(f"🔁 Embedding chunk ({len(text)} chars)...")
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId=args.bedrock_model_id,
        contentType="application/json",
        accept="application/json",
        body=body
    )
    result = json.loads(response['body'].read())
    print("✅ Embedding received")
    return result['embedding']

# --- Indexing Utility ---
def index_chunk(chunk, vector, source):
    """
    @brief Store an embedded chunk into AOSS index
    @param chunk Text chunk
    @param vector Embedding vector
    @param source Original S3 object key
    """
    print(f"🧠 Indexing chunk from {source}...")
    url = f"{args.aoss_endpoint}/{args.index}/_doc"
    headers = {"Content-Type": "application/json"}
    payload = {
        "chunk": chunk,
        "embedding": vector,
        "source": source
    }
    r = requests.post(url, auth=awsauth, headers=headers, json=payload)
    if r.status_code not in [200, 201]:
        print(f"❌ Failed to index chunk: HTTP {r.status_code}")
        print(r.text)
    else:
        print("✅ Chunk indexed")

# --- Main Processing ---
def process_files():
    """
    @brief Process all text files in S3 prefix: chunk, embed, and index them.
    """
    print(f"📡 Scanning S3 bucket s3://{args.bucket}/{args.prefix}")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=args.bucket, Prefix=args.prefix)

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".txt"):
                continue

            print(f"\n📄 Downloading file: {key}")
            s3_obj = s3.get_object(Bucket=args.bucket, Key=key)
            text = s3_obj["Body"].read().decode("utf-8")
            print(f"📦 File size: {len(text)} characters")

            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 20:
                    print(f"⚠️ Skipping short chunk {i}")
                    continue
                try:
                    print(f"\n➡️ Processing chunk {i+1}/{len(chunks)}")
                    embedding = embed_text(chunk)
                    index_chunk(chunk, embedding, key)
                    time.sleep(0.2)  # Slow down write rate
                except Exception as e:
                    print(f"❌ Exception while processing chunk {i}: {e}")

# --- Entrypoint ---
if __name__ == "__main__":
    print("🚀 Starting RAG embedding and indexing pipeline...")
    process_files()
    print("🏁 All files processed.")

