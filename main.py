import streamlit as st
import os

# Page setup
st.set_page_config(page_title="Healthcare Diagnostic Assistant", layout="wide")

from utils.logger import logger  # Importing logger from your utils module

# Global styling
st.markdown("""
    <style>
    /* Sidebar Logo */
    [data-testid="stSidebar"] {
        background-repeat: no-repeat;
        background-position: top center;
        background-size: contain;  /* Ensures the image doesn't stretch */
        padding-top: 20px;  /* Space from top */
    }
    /* Chat Message Background */
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    /* Main Area */
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Import modules
from app import chatbot_view, differential_diagnosis, pubmed_screener

# Initialize session state variables
def init_session_state():
    defaults = {
        "uploaded_file_name": None,
        "full_text": None,
        "summary": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Sidebar Image
sidebar_image = "assets/healthcare.jpeg"  # Update with the correct image path
st.sidebar.image(sidebar_image, use_column_width=True)  # Replacing use_column_width with use_container_width

# Sidebar navigation
st.sidebar.title("👩‍⚕️ Medical Guide")
st.sidebar.markdown("""
    Small AI companion for your healthcare diagnosis journey.
""")

# Sidebar radio buttons for navigation
page = st.sidebar.radio(
    "Go to",
    ("🧠 Chatbot Assistant", "🩺 Differential Diagnosis", "📚 PubMed Screener")
)

# Logging the selected page
logger.info(f"Page selected: {page}")

# Main area - Handle page selection and logging
try:
    if page == "🧠 Chatbot Assistant":
        logger.info("Loading Chatbot Assistant module...")
        chatbot_view.show()
        logger.info("Chatbot Assistant module loaded successfully.")
    elif page == "🩺 Differential Diagnosis":
        logger.info("Loading Differential Diagnosis module...")
        differential_diagnosis.show()
        logger.info("Differential Diagnosis module loaded successfully.")
    elif page == "📚 PubMed Screener":
        logger.info("Loading PubMed Screener module...")
        pubmed_screener.show()
        logger.info("PubMed Screener module loaded successfully.")
    else:
        logger.warning(f"Unknown page selected: {page}")

except Exception as e:
    logger.error(f"Error occurred while loading the module {page}: {e}")
    st.error(f"⚠️ An error occurred while loading the module: {e}")

