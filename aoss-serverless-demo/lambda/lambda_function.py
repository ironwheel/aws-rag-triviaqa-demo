"""
@file lambda_function.py
@brief AWS Lambda handler for RAG-based question answering using Bedrock and AOSS

@copyright Robert E. Taylor, Extropic Systems, 2025
@license MIT
"""

import json
import boto3
import os
import requests
from requests_aws4auth import AWS4Auth

# --- Configuration from Environment ---
REGION = "us-east-1"
INDEX_NAME = os.environ["INDEX_NAME"]
AOSS_ENDPOINT = os.environ["AOSS_ENDPOINT"]
EMBED_MODEL_ID = "amazon.titan-embed-text-v1"

# --- AWS Auth and Clients ---
session = boto3.Session(region_name=REGION)
credentials = session.get_credentials().get_frozen_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, REGION, "aoss", session_token=credentials.token)
bedrock = session.client("bedrock-runtime")

def embed_text(text):
    """
    @brief Generate embedding vector for a given text using Titan Embed model
    @param text Input question text
    @return List of float embeddings
    """
    body = json.dumps({"inputText": text})
    res = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    return json.loads(res["body"].read())["embedding"]

def knn_search(vector, k=5):
    """
    @brief Query AOSS index using k-NN vector search
    @param vector Embedding vector
    @param k Number of top results
    @return List of retrieved context chunks
    """
    url = f"{AOSS_ENDPOINT}/{INDEX_NAME}/_search"
    headers = {"Content-Type": "application/json"}
    query = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": vector,
                    "k": k
                }
            }
        }
    }
    response = requests.post(url, auth=awsauth, headers=headers, json=query)
    if response.status_code != 200:
        raise Exception(f"AOSS search failed: {response.status_code}\n{response.text}")
    hits = response.json().get("hits", {}).get("hits", [])
    return [hit["_source"]["chunk"] for hit in hits]

def generate_answer(prompt, model_id):
    """
    @brief Call Bedrock model (Claude, Titan, Mistral, or Cohere) to generate a response
    @param prompt Full text prompt to send
    @param model_id Bedrock model ID
    @return Generated response string
    """
    if model_id.startswith("anthropic.claude"):
        print("\U0001F9E0 Using Anthropic Claude model...")
        body = {
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
            "max_tokens_to_sample": 512,
            "temperature": 0.5,
            "top_k": 250,
            "top_p": 0.9,
            "stop_sequences": ["\n\nHuman:"]
        }

    elif model_id.startswith("amazon.titan-text"):
        print("\U0001F9E0 Using Amazon Titan Text model...")
        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0.5,
                "topP": 0.9
            }
        }

    elif model_id.startswith("mistral.mistral-7b"):
        print("\U0001F9E0 Using Mistral model...")
        body = {
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": []
        }

    elif model_id.startswith("cohere.command-r"):
        print("\U0001F9E0 Using Cohere Command-R model...")
        body = {
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.7,
            "p": 0.9,
            "stop_sequences": [],
            "return_likelihoods": "NONE"
        }

    else:
        raise ValueError(f"❌ Unsupported model_id: {model_id}")

    response = bedrock.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    result = json.loads(response["body"].read())

    if model_id.startswith("anthropic.claude"):
        return result["completion"]
    elif model_id.startswith("amazon.titan-text"):
        return result["results"][0]["outputText"]
    elif model_id.startswith("mistral.mistral-7b"):
        return result["outputs"][0]["text"]
    elif model_id.startswith("cohere.command-r"):
        return result["generations"][0]["text"]

def lambda_handler(event, context):
    """
    @brief Lambda entry point for RAG or direct model Q&A
    @param event API Gateway event containing query params
    @param context Lambda execution context
    @return Response with generated answer
    """
    params = event.get("queryStringParameters", {})

    question = params.get("prompt", "")
    model_id = params.get("model_id", "amazon.titan-text-lite-v1")
    use_rag = params.get("enable_rag", "false").lower() == "true"

    if not question:
        return {"statusCode": 400, "body": "Missing 'prompt' query parameter"}

    print("\u261E Question:", question)
    print("\u261E RAG:", use_rag)
    print("\u261E Model ID:", model_id)

    if use_rag:
        vector = embed_text(question)
        context_chunks = knn_search(vector)
        print("\U0001F9E0 Context chunks:", context_chunks)
        full_prompt = f"""Use the context below to answer the question.
Context:
{chr(10).join(context_chunks)}

Question: {question}
Answer:"""
    else:
        full_prompt = question

    print("\u261E Full Prompt:", full_prompt)
    answer = generate_answer(full_prompt, model_id)

    return {
        "statusCode": 200,
        "headers": { "Content-Type": "application/json" },
        "body": json.dumps({"answer": answer.strip()})
    }

