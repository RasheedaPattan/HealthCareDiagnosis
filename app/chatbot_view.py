import sys
import os
# Dynamically add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import streamlit as st
from utils.Loading_reports import process_and_store
from utils.summarize_text import extract_important_details
from app.components.rag_pipeline import get_rag_chain
from utils.logger import logger, log_token_usage  # Import the logger and log_token_usage function


def show():
    st.title("🧠 Chatbot Assistant")
    st.markdown("> Your AI healthcare assistant based on uploaded reports.")

    uploaded_file = st.file_uploader(
        "📄 Upload Patient Report (PDF/Image)", 
        type=["pdf", "png", "jpg", "jpeg"]
    )

    if uploaded_file:
        logger.info("User uploaded a file: %s", uploaded_file.name)
        
        # Create a temp folder if it doesn't exist
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)

        file_path = os.path.join(temp_dir, uploaded_file.name)

        # Only process if a new file is uploaded
        if st.session_state.get("uploaded_file_name") != uploaded_file.name:
            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                # Process and summarize
                logger.info("Processing file: %s", uploaded_file.name)
                full_text = process_and_store(file_path)
                summary = extract_important_details(full_text)

                # Save in session state
                st.session_state.full_text = full_text
                st.session_state.summary = summary
                st.session_state.uploaded_file_name = uploaded_file.name

                logger.info("File processed and summary created for: %s", uploaded_file.name)
            except Exception as e:
                logger.error("Error occurred while processing the file %s: %s", uploaded_file.name, str(e))
                st.error("⚠️ An error occurred while processing the report.")
        else:
            logger.info("File %s already processed, skipping reprocessing.", uploaded_file.name)

    # If we have a summary, show it and allow questions
    if st.session_state.get("summary"):
        st.subheader("📌 Summary of Report")
        st.markdown(st.session_state.summary)

        # Logging download button action
        st.download_button(
            label="📄 Download Summary",
            data=st.session_state["summary"],
            file_name="patient_summary.txt",
            mime="text/plain",
        )
        logger.info("Summary downloaded for file: %s", st.session_state.uploaded_file_name)

        st.markdown("---")

        query = st.text_input("🔎 Ask a medical question based on the report:")

        if query:
            logger.info("User asked a question: %s", query)
            
            # Get the RAG chain and generate the response
            rag_chain = get_rag_chain(st.session_state.full_text)
            response = rag_chain.invoke({"query": query})

            st.markdown("### 🤖 Assistant's Response")
            if isinstance(response, dict) and "result" in response:
                st.markdown(response["result"])
                logger.info("Assistant responded with: %s", response["result"])

                # Log token usage after response is generated
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

            else:
                logger.error("Unexpected response format for query '%s'. Raw response: %s", query, response)
                st.error("⚠️ Unexpected response format. Here's the raw response:")
                st.write(response)
    else:
        logger.info("No summary available. Prompting user to upload a report.")
        st.info("📥 Please upload a patient report to begin.")
