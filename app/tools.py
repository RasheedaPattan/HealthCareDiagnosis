import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from utils.logger import logger, log_token_usage  # Importing logger and log_token_usage function

def get_sql_tool():
    try:
        # Log database URI connection
        db_uri = "sqlite:///data/medical.sqlite"
        logger.info(f"Attempting to connect to database with URI: {db_uri}")
        
        # Initialize SQLDatabase
        logger.info("Initializing SQLDatabase object.")
        db = SQLDatabase.from_uri(db_uri)
        logger.info("Database connection established successfully.")
        
        # Log the creation of the SQLDatabaseToolkit
        logger.info("Creating SQLDatabaseToolkit object.")
        sql_tool = SQLDatabaseToolkit(db=db)
        logger.info("SQLDatabaseToolkit created successfully.")
        
        # Example token usage (for model-generated SQL query or response)
        # If you are using an LLM to generate queries, log tokens here
        if hasattr(sql_tool, "llm"):  # Assuming sql_tool has an LLM to interact with
            # Calculate tokens for the prompt used to generate the SQL query
            prompt = "SELECT * FROM patients WHERE condition='diabetes';"  # Example query
            total_tokens_input = len(prompt.split())  # Rough estimate based on words
            log_token_usage(
                model="gemini-1.5-flash-latest",  # Example model name
                prompt_tokens=total_tokens_input,
                completion_tokens=0,  # No response yet
                total_tokens=total_tokens_input
            )
            
            # Log response token usage if model response is available
            # Assuming `response` is the model's output
            response = "Query executed successfully."  # Example response
            total_tokens_output = len(response.split())
            log_token_usage(
                model="gemini-1.5-flash-latest",  # Example model name
                prompt_tokens=total_tokens_input,
                completion_tokens=total_tokens_output,
                total_tokens=total_tokens_input + total_tokens_output
            )
        
        return sql_tool
    
    except Exception as e:
        # Log the error in case of any failure
        logger.error(f"Error occurred while setting up SQL tool: {e}", exc_info=True)
        raise
