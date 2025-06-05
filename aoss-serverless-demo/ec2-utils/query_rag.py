"""#!/usr/bin/env python3
"""
@file query_rag.py
@brief Command-line RAG query tool using Bedrock and AOSS.

@copyright Robert E. Taylor, Extropic Systems, 2025
@license MIT
"""

import argparse
import boto3
import json
import requests
from requests_aws4auth import AWS4Auth

# --- Command-line Argument Parsing ---
parser = argparse.ArgumentParser(description="Query AOSS index with Titan embedding and Bedrock generation")
parser.add_argument("--region", required=True, help="AWS region")
parser.add_argument("--index", required=True, help="AOSS index name")
parser.add_argument("--aoss-endpoint", required=True, help="OpenSearch Serverless endpoint")
parser.add_argument("--embed-model-id", default="amazon.titan-embed-text-v1", help="Embedding model ID")
parser.add_argument("--gen-model-id", default="amazon.titan-text-lite-v1", help="Text generation model ID")
parser.add_argument("--top-k", type=int, default=5, help="Number of nearest neighbors")
parser.add_argument("--enable-rag", action="store_true", help="Enable RAG (default is off)")
args = parser.parse_args()

# --- AWS Clients and Auth ---
session = boto3.Session(region_name=args.region)
credentials = session.get_credentials().get_frozen_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    args.region,
    "aoss",
    session_token=credentials.token,
)

bedrock = session.client("bedrock-runtime", region_name=args.region)
bedrock_gen = bedrock

# --- Embedding Function ---
def embed_query(text):
    """
    @brief Get Titan embedding for input text
    @param text The input question string
    @return List of float embeddings
    """
    print("🔁 Embedding prompt...")
    payload = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId=args.embed_model_id,
        contentType="application/json",
        accept="application/json",
        body=payload
    )
    result = json.loads(response["body"].read())
    return result["embedding"]

# --- AOSS Search ---
def knn_search(vector):
    """
    @brief Search AOSS index using k-NN
    @param vector Embedding vector
    @return List of top context chunks
    """
    print(f"🔍 Running k-NN search (k={args.top_k})...")
    url = f"{args.aoss_endpoint}/{args.index}/_search"
    headers = {"Content-Type": "application/json"}
    query = {
        "size": args.top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": vector,
                    "k": args.top_k
                }
            }
        }
    }
    response = requests.post(url, auth=awsauth, headers=headers, json=query)
    if response.status_code != 200:
        raise Exception(f"AOSS search failed: {response.status_code}\n{response.text}")
    hits = response.json().get("hits", {}).get("hits", [])
    return [hit["_source"]["chunk"] for hit in hits]

# --- Prompt Builder ---
def build_prompt(user_question, context_chunks):
    """
    @brief Build prompt for generative model
    @param user_question Input question
    @param context_chunks Retrieved context
    @return Full prompt string
    """
    context_block = "\n\n".join(context_chunks)
    return f"""Use the context below to answer the question.

Context:
{context_block}

Question: {user_question}
Answer:"""

# --- Call Text Generator ---
def call_bedrock_generator(prompt):
    """
    @brief Call Titan or Claude with prompt
    @param prompt Full prompt to send
    @return Generated output text
    """
    print("🧠 Generating response from Titan Text...")
    body = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 512,
            "temperature": 0.5,
            "topP": 0.9
        }
    }
    response = bedrock_gen.invoke_model(
        modelId=args.gen_model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    result = json.loads(response["body"].read())
    return result["results"][0]["outputText"]

# --- Main Loop ---
if __name__ == "__main__":
    print("🤖 Titan Text Lite RAG Mode")
    print("Press Ctrl+C to quit\n")

    while True:
        try:
            question = input("Q: ").strip()
            if not question:
                continue

            if args.enable_rag:
                embedding = embed_query(question)
                context = knn_search(embedding)
                prompt = build_prompt(question, context)
            else:
                prompt = question

            print("\n📨 Prompt being sent to Titan:")
            print("-" * 60)
            print(prompt if args.enable_rag else "(RAG disabled) " + prompt)
            print("-" * 60)

            answer = call_bedrock_generator(prompt)
            print("\n💬 Answer:\n" + answer.strip())

        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

