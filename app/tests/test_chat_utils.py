import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest
from unittest.mock import MagicMock, patch
from app.components.chat_utils import ChatAgent
from app.components.prompts import chat_prompt_template

class TestChatAgent(unittest.TestCase):
    def setUp(self):
        # Mock the LLM
        self.mock_llm = MagicMock()
        self.mock_llm.invoke.return_value = {"result": "Mock response"}

        # Create the ChatAgent with mock LLM
        self.agent = ChatAgent(prompt=chat_prompt_template, llm=self.mock_llm)

    @patch("app.components.chat_utils.st")
    def test_display_messages_empty_history(self, mock_st):
        # Mock streamlit chat message
        mock_st.chat_message.return_value.write = MagicMock()

        # Clear any previous messages
        self.agent.history.clear()

        # Display messages
        self.agent.display_messages()

        # Check that a default assistant message was added
        self.assertEqual(self.agent.history.messages[0].content, "How can I help you?")
        mock_st.chat_message.assert_called()  # Check Streamlit wrote the message

    @patch("app.components.chat_utils.st")
    def test_start_conversation_flow(self, mock_st):
        # Mock chat input to simulate user typing
        mock_st.chat_input.return_value = "What is LangChain?"

        # Mock chat message writing
        mock_st.chat_message.return_value.write = MagicMock()

        # Mock session state to avoid streamlit session warnings
        mock_st.session_state = {}

        # Start conversation
        self.agent.start_conversation()

        # Check if invoke was called
        self.mock_llm.invoke.assert_called_once()

        # Validate that the correct input was passed
        args, kwargs = self.mock_llm.invoke.call_args
        self.assertIn("question", args[0])
        self.assertEqual(args[0]["question"], "What is LangChain?")

if __name__ == "__main__":
    unittest.main()
