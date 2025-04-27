import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import logging
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from utils.embed_store import get_vectorstore
from utils.logger import logger, log_token_usage  # Import the log_token_usage function

def get_rag_chain(context_text):
    try:
        logger.info("Starting the RAG chain process.")

        # Create vectorstore and retriever
        logger.info("Initializing vectorstore and retriever.")
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        logger.info("Vectorstore and retriever initialized successfully.")

        # Initialize LLM
        logger.info("Initializing ChatGoogleGenerativeAI LLM.")
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-flash-latest",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.3
        )
        logger.info("LLM initialized successfully.")

        # System prompt for RAG model
        system_prompt = """
        You are a highly intelligent, professional, and empathetic medical assistant designed to support healthcare providers...
        --- Context for medical questions ---
        Patient Report:
        {context}
        --- Provider's Question ---
        {question}
        """
        logger.info("System prompt prepared successfully.")

        # Prepare prompt template and chain
        prompt = PromptTemplate(
            template=system_prompt,
            input_variables=["context", "question"]
        )

        from langchain.chains import LLMChain
        logger.info("Creating the LLMChain.")
        chain = LLMChain(
            llm=llm,
            prompt=prompt
        )
        logger.info("LLMChain created successfully.")

        # Wrap output with 'result' key to avoid Streamlit rendering issues
        def answer_with_context(inputs):
            try:
                logger.info("Invoking the RAG chain with user query.")
                response = chain.invoke({
                    "context": context_text,
                    "question": inputs["query"]
                })
                logger.info("Response generated successfully.")
                
                # Log token usage
                if hasattr(response, 'usage'):
                    prompt_tokens = response.usage.get('prompt_tokens', 0)
                    completion_tokens = response.usage.get('completion_tokens', 0)
                    total_tokens = prompt_tokens + completion_tokens
                    log_token_usage(
                        model="gemini-1.5-flash-latest",  # Model name
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens
                    )

                # Return the result in a format that avoids Streamlit rendering issues
                if isinstance(response, dict) and "text" in response:
                    return {"result": response["text"]}
                return {"result": str(response)}

            except Exception as e:
                logger.error(f"Error during response generation: {str(e)}")
                return {"result": f"⚠️ Error generating response: {str(e)}"}

        from langchain_core.runnables import RunnableLambda
        logger.info("Returning the RunnableLambda function.")
        return RunnableLambda(answer_with_context)

    except Exception as e:
        logger.error(f"Failed to create RAG chain: {str(e)}")
        raise e
