import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.extract_text import extract_text
from utils.embed_store import store_embeddings
from utils.logger import logger, log_token_usage  # Import log_token_usage from logger.py file

def process_and_store(file_path):
    logger.info(f"Starting to process and store the report from file: {file_path}")
    
    # Extract text from the uploaded file
    try:
        text = extract_text(file_path)
        logger.info(f"Successfully extracted text from file: {file_path}")
        
        # Estimate token usage for extracted text
        tokens = len(text.split())  # Estimate token count based on word count
        log_token_usage(
            model="text-extraction",  # Custom model name for text extraction
            prompt_tokens=tokens,
            completion_tokens=0,  # No completion tokens for extraction
            total_tokens=tokens
        )
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return ""  # Return empty string in case of error

    # Store the embeddings in the vector store
    try:
        store_embeddings(text)
        logger.info(f"Successfully stored embeddings for the file: {file_path}")
        
        # Estimate token usage for embeddings
        tokens = len(text.split())  # Tokens used for embeddings (similar to text extraction)
        log_token_usage(
            model="embedding-storage",  # Custom model name for embedding storage
            prompt_tokens=tokens,
            completion_tokens=0,  # No completion tokens for embedding storage
            total_tokens=tokens
        )
    except Exception as e:
        logger.error(f"Error storing embeddings for {file_path}: {e}")

    return text
