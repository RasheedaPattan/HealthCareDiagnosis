import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from langchain.memory import ConversationBufferMemory
from utils.logger import logger, log_token_usage  # Importing the logger and log_token_usage function

def get_memory():
    logger.info("Creating new conversation memory.")
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    logger.info("Conversation memory created successfully.")
    
    # Track token usage (as the memory itself doesn't directly track token usage)
    def log_memory_usage(messages):
        """
        Function to log the token usage each time the memory is updated.
        This assumes each message is passed through a language model.
        """
        try:
            # Assuming `messages` are passed to a language model (like ChatGoogleGenerativeAI) for processing
            # If this is part of a model chain, you can use the model's response to track tokens
            total_tokens = sum(len(message['content'].split()) for message in messages)  # A rough estimate of token count
            log_token_usage(
                model="gemini-1.5-flash-latest",  # Model name
                prompt_tokens=total_tokens,  # For simplicity, using total tokens for now
                completion_tokens=0,  # Completion tokens would be tracked elsewhere
                total_tokens=total_tokens
            )
            logger.info(f"Token usage logged: {total_tokens} tokens used.")
        except Exception as e:
            logger.error(f"Error during token usage logging: {str(e)}")
    
    # You can call `log_memory_usage` whenever the memory is updated (this depends on how you use the memory)
    memory.add_message = lambda message: (log_memory_usage([message]), memory.add_message(message))
    
    return memory
