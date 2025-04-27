# differential_diagnosis.py

import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.components.rag_pipeline import get_rag_chain
from utils.logger import logger, log_token_usage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

load_dotenv()

def show():
    st.title("🩻 Differential Diagnosis Assistant")

    if "full_text" not in st.session_state or not st.session_state.full_text:
        st.warning("⚠️ Please upload a report in the Chatbot tab first.")
        logger.warning("User attempted to access Differential Diagnosis without uploading a report.")
        return

    # ⚡ FIX: Detect new file and regenerate
    if "previous_full_text" not in st.session_state or st.session_state.full_text != st.session_state.previous_full_text:
        logger.info("New uploaded report detected. Clearing old diagnosis...")

        if "diagnostic_summary" in st.session_state:
            del st.session_state.diagnostic_summary

        st.session_state.previous_full_text = st.session_state.full_text

    # Generate differential diagnosis
    if "diagnostic_summary" not in st.session_state:
        logger.info("Generating new diagnostic summary...")

        prompt_template = PromptTemplate.from_template("""
You're a top-tier diagnostic AI trained on complex medical cases. A healthcare provider has uploaded a detailed patient report. Based solely on this report and without assumptions, provide a ranked list of **differential diagnoses** with reasoning.

Apart from the diagnosis extracted from the patient report you will be able to provide other possible different diagnosis by understanding minute details of that report.

Report:
{report_text}

Output format:
1. **Primary Diagnosis**: [Diagnosis]  
   Reasoning: [Detailed clinical rationale]  

2. **Secondary Possibility**: [Diagnosis]  
   Reasoning: [...]

3. **Less Likely but Consider**: [Diagnosis]  
   Reasoning: [...]

Also mention if any urgent/life-threatening conditions only if it is needed and must be ruled out and why.
""")

        prompt = prompt_template.format(report_text=st.session_state.full_text)

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.4,
            google_api_key=os.getenv("GEMINI_API_KEY")  # Make sure your .env has GEMINI_API_KEY
        )

        try:
            response = llm.invoke(prompt)
            st.session_state.diagnostic_summary = response.content
            logger.info("✅ Diagnostic summary generated successfully.")

            # Log token usage
            if hasattr(response, 'usage'):
                prompt_tokens = response.usage.get('prompt_tokens', 0)
                completion_tokens = response.usage.get('completion_tokens', 0)
                total_tokens = prompt_tokens + completion_tokens
                log_token_usage(
                    model="gemini-1.5-flash-latest",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens
                )

        except Exception as e:
            logger.error(f"❌ Error generating diagnostic summary: {str(e)}")
            st.error("⚠️ Error generating differential diagnosis.")
            return

    # Display Diagnostic Summary
    st.markdown("### 🔬 Diagnostic Reasoning")
    st.markdown(st.session_state.diagnostic_summary)

    # Chatbot based on Diagnosis
    st.markdown("---")
    st.subheader("💬 Ask about the Differential Diagnosis")

    query = st.text_input("Ask a question related to the diagnosis...")

    if query:
        logger.info(f"User asked: {query}")

        try:
            combined_context = st.session_state.full_text + "\n\n" + st.session_state.diagnostic_summary
            rag_chain = get_rag_chain(combined_context)
            response = rag_chain.invoke({"query": query})

            st.markdown("### 🤖 Assistant's Response")

            if isinstance(response, dict) and "result" in response:
                st.markdown(response["result"])
                logger.info(f"Assistant responded: {response['result']}")

                # Log token usage if available
                if hasattr(response, 'usage'):
                    prompt_tokens = response.usage.get('prompt_tokens', 0)
                    completion_tokens = response.usage.get('completion_tokens', 0)
                    total_tokens = prompt_tokens + completion_tokens
                    log_token_usage(
                        model="gemini-1.5-flash-latest",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens
                    )
            else:
                logger.warning("⚠️ Unexpected response format.")
                st.write(response)

        except Exception as e:
            logger.error(f"❌ Error processing query: {str(e)}")
            st.error("⚠️ Error processing your question.")
