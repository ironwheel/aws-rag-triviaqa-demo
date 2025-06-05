# Website Folder - README

This folder contains the static frontend for the **RAG TriviaQA Demo** hosted on S3.
It includes a single-page web app (`index.html`) that calls an API Gateway connected to a Lambda function.

---

## 📄 index.html

This is the main UI for the demo. It:

* Allows users to enter a question.
* Offers model selection from a dropdown.
* Provides a checkbox to enable Retrieval-Augmented Generation (RAG).
* Displays results from the backend Bedrock + AOSS pipeline.

It loads configuration from the adjacent `config.json` file to stay environment-agnostic.

---

## ⚙️ config.json Schema

The `config.json` file must be placed in the same S3 folder as `index.html`.
It contains environment-specific and UI configuration values.

```json
{
  "apiEndpoint": "https://<your-api-id>.execute-api.<region>.amazonaws.com/<stage>/<route>",
  "modelOptions": [
    "amazon.titan-text-lite-v1",
    "anthropic.claude-v2",
    "mistral.mistral-7b-instruct",
    "cohere.command-r"
  ],
  "exampleQuestions": [
    "Where in England was Dame Judi Dench born?",
    "From which country did Angola achieve independence in 1975?",
    "Who won Super Bowl XX?",
    "Which William wrote the novel Lord Of The Flies?",
    "... up to 100 items ..."
  ]
}
```

### Properties

* **`apiEndpoint`**: Fully qualified API Gateway URL that invokes the Lambda function
* **`modelOptions`**: List of model IDs supported by the Bedrock runtime
* **`exampleQuestions`**: Optional helper list to populate a collapsible list of questions for users to try

---

## 🌐 Hosting

You can host this website via an [S3 static website](https://docs.aws.amazon.com/AmazonS3/latest/userguide/WebsiteHosting.html). Make sure to enable CORS in your Lambda to allow the S3 domain as origin.

---

For help deploying, see the main project README at the root of this repository.

