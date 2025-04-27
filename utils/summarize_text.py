import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.logger import logger  # Adjust the import path according to your project structure

# Load environment variables
load_dotenv()

# Define prompt template for summarization
summarize_prompt = PromptTemplate.from_template("""
You are a professional medical assistant helping a healthcare provider. Summarize the key information from the patient report below.

Extract and clearly format:
- Patient Name
- Age / Gender
- Symptoms
- Diagnosis / Conditions
- Medications
- Dosage Instructions
- Precautions (Food, Sleep, Activity, Yoga)
- Follow-up Advice

Report:
{report_text}
""")

# Initialize LLM (ChatGoogleGenerativeAI)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.3,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Define the chain for processing the summary
chain: RunnableSequence = summarize_prompt | llm

def log_token_usage(model, prompt_tokens, completion_tokens, total_tokens):
    """
    Logs token usage information including the model, prompt tokens, completion tokens,
    and total tokens to both the console and the log file.
    """
    logger.info(f"Model: {model}, "
                f"Prompt Tokens: {prompt_tokens}, "
                f"Completion Tokens: {completion_tokens}, "
                f"Total Tokens: {total_tokens}")

from langchain.callbacks import get_openai_callback

def extract_important_details(text: str) -> str:
    logger.info("Starting to extract important details from the provided report.")
    
    try:
        logger.info(f"Processing report text of length {len(text)} characters.")
        
        # Start a callback to track token usage
        with get_openai_callback() as cb:
            response = chain.invoke({"report_text": text})

        logger.info(f"Token usage - Prompt Tokens: {cb.prompt_tokens}, "
                    f"Completion Tokens: {cb.completion_tokens}, "
                    f"Total Tokens: {cb.total_tokens}")

        log_token_usage(
            model="gemini-1.5-flash-latest",
            prompt_tokens=cb.prompt_tokens,
            completion_tokens=cb.completion_tokens,
            total_tokens=cb.total_tokens,
        )

        return response.content
    
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        return f"⚠️ Error during summarization: {e}"

