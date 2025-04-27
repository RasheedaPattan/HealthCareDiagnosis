import logging
import os
import csv
from datetime import datetime

# 📁 Create 'logs' directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Paths inside logs folder
LOG_FILE_PATH = os.path.join(log_dir, "healthcare_assistant.log")
TOKEN_USAGE_FILE = os.path.join(log_dir, "token_usage.csv")

# Set up a logger
def setup_logger():
    """
    Sets up the logger for the Healthcare Diagnostic Assistant.
    Logs information to both the console and a log file.
    """
    logger = logging.getLogger('HealthcareDiagnosticAssistant')
    logger.setLevel(logging.DEBUG)  # Log everything from DEBUG level and above

    # Avoid adding handlers multiple times if already added
    if not logger.handlers:
        # Create file handler to log into a file
        file_handler = logging.FileHandler(LOG_FILE_PATH)
        file_handler.setLevel(logging.DEBUG)

        # Create console handler for real-time logging
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Define log formatting
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Create logger instance
logger = setup_logger()

# Token usage logging function
def log_token_usage(model, prompt_tokens, completion_tokens, total_tokens):
    """
    Logs token usage information to a CSV file and also logs it to the console and log file.
    """
    # Log to console + log file
    logger.info(f"Model: {model}, Prompt Tokens: {prompt_tokens}, "
                f"Completion Tokens: {completion_tokens}, Total Tokens: {total_tokens}")

    # Save token usage into CSV
    file_exists = os.path.isfile(TOKEN_USAGE_FILE)

    try:
        with open(TOKEN_USAGE_FILE, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists:
                # Write CSV header if file doesn't exist
                writer.writerow(["timestamp", "model", "prompt_tokens", "completion_tokens", "total_tokens"])

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model,
                prompt_tokens,
                completion_tokens,
                total_tokens
            ])
    except Exception as e:
        logger.error(f"Failed to log token usage to CSV: {e}")
