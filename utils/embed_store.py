import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.logger import logger, log_token_usage  # Importing logger and log_token_usage function

# Load environment variables
load_dotenv()

def store_embeddings(text, source_name="uploaded_report", persist_directory="data/chroma_db/default"):
    try:
        logger.info(f"Starting to store embeddings for source: {source_name}")
        
        # Split the text into chunks
        logger.info("Splitting the text into chunks.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.create_documents([text])
        logger.info(f"Text split into {len(docs)} chunks.")

        # Attach metadata to each chunk
        for doc in docs:
            doc.metadata["source"] = source_name

        # Set up embeddings
        logger.info("Initializing embeddings.")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # Track token usage for embeddings creation
        total_input_tokens = sum(len(doc.page_content.split()) for doc in docs)
        log_token_usage(
            model="models/embedding-001",  # Embedding model
            prompt_tokens=total_input_tokens,
            completion_tokens=0,  # No response yet
            total_tokens=total_input_tokens
        )
        
        # Save to Chroma
        logger.info(f"Saving embeddings to ChromaDB at {persist_directory}.")
        os.makedirs(persist_directory, exist_ok=True)
        db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
        db.persist()
        logger.info(f"Successfully stored {len(docs)} chunks to ChromaDB.")
        
        return f"Stored {len(docs)} chunks to ChromaDB at {persist_directory}"

    except Exception as e:
        logger.error(f"Error storing embeddings: {str(e)}")
        return f"⚠️ Error: {str(e)}"

def get_vectorstore(persist_directory="data/chroma_db/default"):
    try:
        logger.info("Initializing embeddings for vectorstore retrieval.")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        # Track token usage for loading and embedding creation for retrieval
        logger.info(f"Loading ChromaDB from {persist_directory}.")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
        # Estimate token usage for vectorstore retrieval
        retrieval_tokens = 0  # You might need a custom way to estimate tokens used during retrieval (if applicable)
        log_token_usage(
            model="models/embedding-001",  # Embedding model
            prompt_tokens=0,  # No new prompt
            completion_tokens=retrieval_tokens,
            total_tokens=retrieval_tokens
        )
        
        logger.info("Successfully loaded vectorstore.")
        return vectorstore

    except Exception as e:
        logger.error(f"Error retrieving vectorstore: {str(e)}")
        raise e
