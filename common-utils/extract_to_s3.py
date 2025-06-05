#!/usr/bin/env python3
"""
@file extract_to_s3.py
@brief Extracts Wikipedia EntityPages from TriviaQA archive and uploads them to S3
@details
    - Extracts relevant .txt files listed in wikipedia-train.json from a .tar.gz archive
    - Uploads them to an S3 bucket under evidence/wikipedia/
    - Ensures no duplicate uploads or extractions

@copyright Robert E. Taylor, Extropic Systems, 2025
@license MIT
"""

import argparse
import json
import subprocess
from pathlib import Path
import boto3

def extract_and_upload(json_path, archive_path, output_dir, bucket, count):
    """
    @brief Extracts and uploads files referenced in the JSON metadata.
    @param json_path Path to wikipedia-train.json
    @param archive_path Path to the TriviaQA tar.gz archive
    @param output_dir Local directory to extract evidence files to
    @param bucket S3 bucket name
    @param count Number of records to process
    """
    s3 = boto3.client("s3")
    extracted_files = set()
    uploaded_files = set()

    with open(json_path, 'r') as f:
        data = json.load(f)["Data"]

    for idx, record in enumerate(data[:count]):
        question = record.get("Question", "[No Question]")
        print(f"\nRecord {idx + 1}: {question}")

        entity_pages = record.get("EntityPages", [])
        for doc in entity_pages:
            filename = doc["Filename"]
            archive_member = f"evidence/wikipedia/{filename}"
            local_path = Path(output_dir) / archive_member
            s3_key = f"evidence/wikipedia/{filename}"

            if archive_member in extracted_files:
                continue

            if not local_path.exists():
                local_path.parent.mkdir(parents=True, exist_ok=True)
                extract_command = [
                    "tar", "-xf", archive_path,
                    "-C", output_dir,
                    archive_member
                ]

                try:
                    subprocess.run(extract_command, check=True)
                    print(f"  ✅ Extracted: {archive_member}")
                    extracted_files.add(archive_member)
                except subprocess.CalledProcessError as e:
                    print(f"  ❌ Failed to extract {archive_member}: {e}")
                    continue
            else:
                print(f"  ⏩ Skipping extract (already exists): {local_path}")
                extracted_files.add(archive_member)

            if s3_key in uploaded_files:
                continue

            try:
                s3.upload_file(str(local_path), bucket, s3_key)
                print(f"  ☁️ Uploaded to S3: {s3_key}")
                uploaded_files.add(s3_key)
            except Exception as e:
                print(f"  ❌ Failed to upload {s3_key}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts TriviaQA Wikipedia EntityPages and uploads them to S3.")
    parser.add_argument("--json-path", required=True, help="Path to wikipedia-train.json")
    parser.add_argument("--archive-path", required=True, help="Path to .tar.gz archive")
    parser.add_argument("--output-dir", required=True, help="Local directory to extract files into")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--count", type=int, default=10, help="Number of records to process")

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    extract_and_upload(args.json_path, args.archive_path, args.output_dir, args.bucket, args.count)

