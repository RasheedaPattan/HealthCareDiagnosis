import sys
import os
import logging

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import streamlit as st
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.base import Runnable
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

# Use the logger from logger.py (assuming it's already set up in your project)
from utils.logger import logger, log_token_usage  # Import the log_token_usage function

class ChatAgent:
    def __init__(self, prompt: ChatPromptTemplate, llm: Runnable):
        """
        Initialize the ChatAgent.
        """
        self.history = StreamlitChatMessageHistory(key="chat_history")
        self.llm = llm
        self.prompt = prompt
        self.chain = self.setup_chain()
        logger.info("ChatAgent initialized with prompt and llm.")

    def setup_chain(self) -> RunnableWithMessageHistory:
        """
        Set up the chain for the ChatAgent.
        """
        logger.info("Setting up the chain for the ChatAgent.")
        chain = self.prompt | self.llm
        logger.info("Chain setup complete.")
        return RunnableWithMessageHistory(
            chain,
            lambda session_id: self.history,
            input_messages_key="question",  # This must match your prompt input
            history_messages_key="history"  # Must match MessagesPlaceholder name
        )

    def display_messages(self):
        logger.info("Displaying chat messages.")
        if len(self.history.messages) == 0:
            self.history.add_ai_message("How can I help you today?")

        with st.container():
            for msg in self.history.messages:
                if msg.type == "human":
                    with st.chat_message("human"):
                        st.markdown(f"**You:** {msg.content}")
                else:
                    with st.chat_message("ai"):
                        st.markdown(f"**Assistant:** {msg.content}")

    def start_conversation(self):
        logger.info("Starting conversation.")
        with st.container():
            self.display_messages()
            user_question = st.chat_input(placeholder="Ask me anything!")

            if user_question:
                logger.info(f"User asked: {user_question}")
                st.chat_message("human").write(user_question)
                config = {"configurable": {"session_id": "session"}}

                with st.spinner("✏️ Thinking..."):
                    try:
                        logger.info("Invoking the chain to get the response.")
                        response = self.chain.invoke({"question": user_question}, config)
                        
                        if hasattr(response, "content"):
                            st.chat_message("ai").write(response.content)
                            logger.info(f"Assistant responded: {response.content}")

                            # Assuming the response contains token usage details
                            # If you have a way to track the tokens used by the model, do it here
                            # Log token usage
                            prompt_tokens = response.get("prompt_tokens", 0)  # Replace with actual token tracking logic
                            completion_tokens = response.get("completion_tokens", 0)  # Replace with actual token tracking logic
                            total_tokens = prompt_tokens + completion_tokens
                            
                            # Log the token usage
                            log_token_usage(
                                model="gemini-1.5-flash-latest",  # Replace with actual model name if different
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                total_tokens=total_tokens
                            )

                        else:
                            st.chat_message("ai").write(str(response))
                            logger.warning(f"Unexpected response format: {response}")
                    except Exception as e:
                        logger.error(f"An error occurred: {e}")
                        st.error(f"An error occurred: {e}")
