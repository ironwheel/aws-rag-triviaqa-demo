#!/usr/bin/env python3
"""
@file query_rag.py
@brief Interactive CLI for querying a FAISS vector index using Bedrock language models with optional RAG.
@details
    - Loads a FAISS vector index and metadata
    - Embeds a user question using Titan
    - Retrieves top-k relevant chunks if RAG is enabled
    - Builds a context-aware or raw prompt
    - Sends prompt to Bedrock model (Claude, Titan, or Mistral)
    - Prints generated answer

@copyright Robert E. Taylor, Extropic Systems, 2025
@license MIT
"""

import faiss
import json
import boto3
import argparse
import numpy as np

# --- Settings ---
EMBEDDING_DIM = 1536
TOP_K = 5

def load_index(index_path):
    """
    @brief Load a FAISS index from disk.
    @param index_path Path to .index file
    @return FAISS index object
    """
    return faiss.read_index(index_path)

def load_metadata(metadata_path):
    """
    @brief Load associated metadata (chunk text and source) from JSON.
    @param metadata_path Path to metadata.json
    @return List of metadata dictionaries
    """
    with open(metadata_path, "r") as f:
        return json.load(f)

def get_embedding(text, bedrock):
    """
    @brief Generate Titan embedding for a question or chunk.
    @param text Input text
    @param bedrock Boto3 Bedrock runtime client
    @return Numpy array of embedding vector
    """
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    result = json.loads(response['body'].read())
    return np.array(result['embedding'], dtype='float32')

def retrieve_top_k(index, embedding, metadata, k=TOP_K):
    """
    @brief Run a top-k vector similarity search over FAISS index.
    @param index FAISS index object
    @param embedding Embedded query vector
    @param metadata Metadata list to match results to
    @param k Number of top chunks to return
    @return List of top metadata chunks
    """
    D, I = index.search(np.expand_dims(embedding, axis=0), k)
    return [metadata[i] for i in I[0]]

def build_prompt(chunks, question):
    """
    @brief Construct a RAG-style prompt using retrieved context.
    @param chunks List of top-k retrieved metadata chunks
    @param question User question
    @return Prompt string to send to FM
    """
    context = "\n\n".join([f"[{i+1}] {c['chunk']}" for i, c in enumerate(chunks)])
    return (
        f"You are a helpful assistant. Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )

def call_bedrock_model(prompt, bedrock, model_id):
    """
    @brief Submit a prompt to the selected Bedrock model and extract the response.
    @param prompt Full prompt string
    @param bedrock Boto3 Bedrock runtime client
    @param model_id Bedrock model identifier (e.g., Claude, Titan, Mistral)
    @return Response string
    """
    if model_id.startswith("anthropic.claude"):
        body = json.dumps({
            "prompt": prompt,
            "max_tokens_to_sample": 300,
            "temperature": 0.7,
            "stop_sequences": ["\n\n"]
        })
    elif model_id.startswith("amazon.titan-text"):
        body = json.dumps({
            "inputText": prompt
        })
    elif model_id.startswith("mistral."):
        body = json.dumps({
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.7
        })
    else:
        raise ValueError(f"Unsupported model: {model_id}")

    response = bedrock.invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json"
    )

    response_body = json.loads(response['body'].read())

    if model_id.startswith("anthropic.claude"):
        return response_body.get("completion", "[No output]")
    elif model_id.startswith("amazon.titan-text"):
        return response_body.get("results", [{}])[0].get("outputText", "[No output]")
    elif model_id.startswith("mistral."):
        return response_body.get("outputs", [{}])[0].get("text", "[No output]")
    else:
        return "[Unknown model output format]"

def main():
    """
    @brief Main entry point. Loads index, runs an interactive Q&A loop, and prints model responses.
    """
    parser = argparse.ArgumentParser(description="Query a FAISS RAG index using Bedrock")
    parser.add_argument("--index-path", default="faiss.index", help="Path to FAISS index file")
    parser.add_argument("--metadata-path", default="metadata.json", help="Path to metadata JSON")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--model-id", default="anthropic.claude-v2", help="Bedrock model ID to use")
    parser.add_argument("--disable-rag", action="store_true", help="Disable RAG and send only the question to the model")
    args = parser.parse_args()

    # Load index and metadata
    index = load_index(args.index_path)
    metadata = load_metadata(args.metadata_path)
    bedrock = boto3.client("bedrock-runtime", region_name=args.region)

    print("✅ FAISS index and metadata loaded. Type your question (or Ctrl+C to exit).")

    while True:
        try:
            question = input("\nQ: ").strip()
            if not question:
                continue

            if args.disable_rag:
                # No context – raw prompt
                prompt = f"Answer the following question:\n\nQuestion: {question}\n\nAnswer:"
            else:
                embedding = get_embedding(question, bedrock)
                top_chunks = retrieve_top_k(index, embedding, metadata)
                print("\n🔍 Top retrieved chunks:")
                for i, c in enumerate(top_chunks):
                    print(f"[{i+1}] {c['source']}: {c['chunk'][:150]}...\n")
                prompt = build_prompt(top_chunks, question)

            answer = call_bedrock_model(prompt, bedrock, model_id=args.model_id)
            print(f"\n🧠 Answer:\n{answer}")

        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break

if __name__ == "__main__":
    main()

