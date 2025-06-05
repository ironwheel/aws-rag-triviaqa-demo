# Lambda Function for RAG Querying

This Lambda is designed to run inside a VPC with private access to:
- OpenSearch Serverless (AOSS)
- Bedrock (Titan / Claude)
- S3 (optional for document lookup)

It is invoked through an **API Gateway (HTTP GET)** request.

## Query Parameters

- `prompt`: The user question
- `model_id`: Bedrock model to invoke (e.g., `amazon.titan-text-lite-v1`)
- `enable_rag`: `true` or `false`

## Sample Event
```json
{
  "queryStringParameters": {
    "prompt": "Where was Judi Dench born?",
    "model_id": "amazon.titan-text-lite-v1",
    "enable_rag": "true"
  }
}
