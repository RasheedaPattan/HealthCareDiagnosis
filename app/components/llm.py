import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Use the logger from logger.py (assuming it's already set up in your project)
from utils.logger import logger, log_token_usage  # Import log_token_usage to log token data

# Load environment variables
logger.info("Loading environment variables from .env file.")
load_dotenv()

# Retrieve Google API key
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
if GOOGLE_API_KEY is None:
    logger.error("GEMINI_API_KEY not found in environment variables!")
else:
    logger.info("Successfully loaded GEMINI_API_KEY from environment.")

# Initialize the LLM
try:
    logger.info("Initializing ChatGoogleGenerativeAI model.")
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest",
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )
    logger.info("ChatGoogleGenerativeAI model initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize ChatGoogleGenerativeAI model: {e}")

# Example function to invoke the model and log token usage
def generate_response(prompt: str):
    try:
        logger.info(f"Generating response for prompt: {prompt}")
        
        # Invoke the model with the prompt as a string (not a dictionary)
        response = llm.invoke(prompt)  # Pass the prompt directly as a string
        
        # Check if response has 'content' (for AIMessage object)
        if hasattr(response, "content"):
            logger.info(f"Response content: {response.content}")

            # Log token usage (we assume the model has a `usage` or similar attribute)
            prompt_tokens = getattr(response, "prompt_tokens", 0)
            completion_tokens = getattr(response, "completion_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens

            # Log the token usage
            log_token_usage(
                model="gemini-1.5-flash-latest",  # Model name
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )

            logger.info(f"Assistant responded: {response.content}")
            return response.content
        else:
            logger.warning(f"Unexpected response format: {response}")
            return "⚠️ Error: Unexpected response format."
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"⚠️ Error: {e}"


