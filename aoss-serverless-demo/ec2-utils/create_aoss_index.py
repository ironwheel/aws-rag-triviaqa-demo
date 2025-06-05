#!/usr/bin/env python3
"""
@file create_index.py
@brief Creates an Amazon OpenSearch Serverless (AOSS) index for vector search using the FAISS engine.
@details
    - Enables k-NN indexing on a specified endpoint
    - Creates fields for chunk text, 1536-dim embeddings, and source info
    - Uses SigV4 signed HTTP request via requests-aws4auth

@copyright Robert E. Taylor, Extropic Systems, 2025
@license MIT
"""

import boto3
import json
import argparse
import requests
from requests_aws4auth import AWS4Auth

def get_awsauth(region: str) -> AWS4Auth:
    """
    @brief Generates AWS SigV4 credentials for requests.
    @param region AWS region (e.g., us-east-1)
    @return AWS4Auth instance for signing HTTP requests
    """
    session = boto3.Session()
    credentials = session.get_credentials().get_frozen_credentials()
    return AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        "aoss",
        session_token=credentials.token,
    )

def create_index(endpoint: str, index_name: str, region: str):
    """
    @brief Sends a signed PUT request to create an AOSS index.
    @param endpoint AOSS collection endpoint (e.g., https://abc123.us-east-1.aoss.amazonaws.com)
    @param index_name Name of the index to create
    @param region AWS region (e.g., us-east-1)
    """
    awsauth = get_awsauth(region)
    index_url = f"{endpoint.rstrip('/')}/{index_name}"
    headers = {"Content-Type": "application/json"}

    payload = {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "chunk": {
                    "type": "text"
                },
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 1536,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "faiss"
                    }
                },
                "source": {
                    "type": "keyword"
                }
            }
        }
    }

    print(f"Creating index '{index_name}' at {index_url}...")
    response = requests.put(index_url, auth=awsauth, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        print("✅ Index created successfully.")
    else:
        print(f"❌ Failed to create index. Status: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an AOSS vector search index using FAISS.")
    parser.add_argument("--endpoint", required=True, help="AOSS endpoint (e.g., https://abc123.us-east-1.aoss.amazonaws.com)")
    parser.add_argument("--index", default="triviaqa", help="Index name to create (default: triviaqa)")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    args = parser.parse_args()

    create_index(args.endpoint, args.index, args.region)

