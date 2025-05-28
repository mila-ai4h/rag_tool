import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# API configuration
API_BASE_URL = "http://localhost:8000"  # Adjust if needed


def get_api_key(args_api_key: Optional[str] = None) -> str:
    """Get API key from arguments or environment variable."""
    api_key = args_api_key or os.getenv("API_KEY")
    if not api_key:
        raise ValueError(
            "API key must be provided either via --api-key argument or API_KEY environment variable"
        )
    return api_key


def delete_collection(name: str) -> bool:
    """Delete a collection if it exists."""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/collections/{name}", headers=HEADERS
        )
        if response.status_code == 200:
            logger.info(f"Collection '{name}' deleted successfully")
            return True
        elif response.status_code == 404:
            logger.info(f"Collection '{name}' does not exist")
            return True
        else:
            logger.error(f"Failed to delete collection '{name}': {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error deleting collection '{name}': {str(e)}")
        return False


def create_collection(name: str) -> bool:
    """Create a new collection."""
    try:
        response = requests.post(f"{API_BASE_URL}/collections/{name}", headers=HEADERS)
        if response.status_code == 200:
            logger.info(f"Collection '{name}' created successfully")
            return True
        elif response.status_code == 409:
            logger.info(f"Collection '{name}' already exists")
            return True
        else:
            logger.error(f"Failed to create collection '{name}': {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error creating collection '{name}': {str(e)}")
        return False


def index_pdf(
    collection: str, file_path: str, source_id: str, tags: str, extras: str
) -> bool:
    """Index a PDF file into the collection."""
    try:
        if not os.path.exists(file_path):
            logger.error(f"PDF file not found: {file_path}")
            return False

        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "application/pdf")}
            data = {
                "source_id": source_id,
                "tags": tags,
            }
            if extras:  # Only add extras if it's not empty
                data["extras"] = extras

            response = requests.post(
                f"{API_BASE_URL}/collections/{collection}/add-pdf",
                headers=HEADERS,
                files=files,
                data=data,
            )

            if response.status_code == 200:
                logger.info(f"Successfully indexed PDF: {file_path}")
                return True
            else:
                logger.error(f"Failed to index PDF {file_path}: {response.text}")
                return False
    except Exception as e:
        logger.error(f"Error indexing PDF {file_path}: {str(e)}")
        return False


def index_url(
    collection: str, url: str, source_id: str, tags: str, extras: str
) -> bool:
    """Index a URL into the collection."""
    try:
        data = {
            "url": url,
            "source_id": source_id,
            "tags": tags,
        }
        if extras:  # Only add extras if it's not empty
            data["extras"] = extras

        response = requests.post(
            f"{API_BASE_URL}/collections/{collection}/add-url",
            headers=HEADERS,
            data=data,
        )

        if response.status_code == 200:
            logger.info(f"Successfully indexed URL: {url}")
            return True
        else:
            logger.error(f"Failed to index URL {url}: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error indexing URL {url}: {str(e)}")
        return False


def process_csv(csv_path: str, collection: str) -> bool:
    """Process the CSV file and index documents."""
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            required_columns = {
                "document_type",
                "file_path",
                "source_id",
                "tags",
                "link",
            }

            # Validate CSV columns
            if not all(col in reader.fieldnames for col in required_columns):
                missing = required_columns - set(reader.fieldnames)
                logger.error(f"Missing required columns in CSV: {missing}")
                return False

            success_count = 0
            error_count = 0

            for row_num, row in enumerate(
                reader, start=2
            ):  # start=2 because row 1 is header
                doc_type = row["document_type"].lower()
                source_id = row["source_id"]
                tags = row["tags"]
                link = row["link"].strip() if row["link"] else ""

                # Create extras JSON from link only if link is not empty
                extras = json.dumps({"link": link}) if link else ""

                # Validate document type
                if doc_type not in ["pdf", "url"]:
                    logger.error(f"Line {row_num}: Invalid document type: {doc_type}")
                    error_count += 1
                    continue

                success = False
                if doc_type == "pdf":
                    success = index_pdf(
                        collection, row["file_path"], source_id, tags, extras
                    )
                else:  # url
                    success = index_url(
                        collection, row["file_path"], source_id, tags, extras
                    )

                if success:
                    success_count += 1
                else:
                    error_count += 1

            logger.info(
                f"Processing complete. Success: {success_count}, Errors: {error_count}"
            )
            return error_count == 0

    except Exception as e:
        logger.error(f"Error processing CSV file: {str(e)}")
        return False


def main():
    """Main function to run the provisioning script."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Provision a collection with documents from a CSV file."
    )
    parser.add_argument("collection_name", help="Name of the collection to create")
    parser.add_argument(
        "csv_path", help="Path to the CSV file containing document information"
    )
    parser.add_argument(
        "--api-key",
        help="API key for authentication (can also be set via API_KEY environment variable)",
    )
    parser.add_argument(
        "--api-base-url",
        default="http://localhost:8000",
        help="Base URL for the API (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    try:
        global API_BASE_URL, HEADERS
        API_BASE_URL = args.api_base_url
        api_key = get_api_key(args.api_key)
        HEADERS = {"X-API-Key": api_key}
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    collection_name = args.collection_name
    csv_path = args.csv_path

    # Ensure CSV file exists
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

    # Delete existing collection
    if not delete_collection(collection_name):
        logger.error("Failed to delete existing collection")
        sys.exit(1)

    # Create new collection
    if not create_collection(collection_name):
        logger.error("Failed to create new collection")
        sys.exit(1)

    # Process CSV file
    if not process_csv(csv_path, collection_name):
        logger.error("Failed to process all documents from CSV")
        sys.exit(1)

    logger.info("Provisioning completed successfully")


if __name__ == "__main__":
    main()
