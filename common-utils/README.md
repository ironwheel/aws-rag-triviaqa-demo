# Common Utilities

This folder contains scripts used to preprocess and upload data to S3 for use in downstream RAG pipelines.

## Files

- **extract_to_s3.py**: Parses the `wikipedia-train.json` metadata file and extracts matching text files from the `triviaqa-rc.tar.gz` archive. The extracted files are uploaded to a specified S3 bucket, preserving the folder structure.

## Usage
```bash
python extract_to_s3.py --metadata-path qa/wikipedia-train.json \
                         --archive-path triviaqa-rc.tar.gz \
                         --s3-bucket triviaqa \
                         --prefix evidence/wikipedia/ \
                         --limit 100

