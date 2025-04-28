import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Use the logger from logger.py 
from utils.logger import logger, log_token_usage  # Import log_token_usage to log token data

# Log the start of the setup process
logger.info("Initializing ChatPromptTemplate with system and human prompts.")

try:
    # Create the ChatPromptTemplate
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an AI chatbot having a conversation with a human."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    logger.info("ChatPromptTemplate initialized successfully.")

except Exception as e:
    # Log any error that occurs during the initialization
    logger.error(f"Failed to initialize ChatPromptTemplate: {e}")

# Example function to track token usage during model invocation
def generate_response_and_log_tokens(question: str):
    try:
        # Assuming you have an LLM (like a generative model) to generate the response
        # This can be done with ChatPromptTemplate or directly with a language model

        # Sample response 
        response = "Sample response to the question." 

        # Simulate token usage (you can get this from the model's response)
        prompt_tokens = 50  # Replace this with the actual token count from your model
        completion_tokens = 150  # Replace with actual token count
        total_tokens = prompt_tokens + completion_tokens

        # Log token usage
        log_token_usage(
            model="gemini-1.5-flash-latest",  # Model name 
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )

        logger.info(f"Response generated: {response}")
        return response

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"⚠️ Error: {e}"

# Example usage
question = "What are the symptoms of diabetes?"
response = generate_response_and_log_tokens(question)
