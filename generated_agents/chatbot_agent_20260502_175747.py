import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

class Agent:
    """
    A simple chatbot agent that uses Google Gemini to generate responses.
    """
    def __init__(self):
        try:
            # Ensure the GOOGLE_API_KEY environment variable is set
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError("GOOGLE_API_KEY environment variable not set.")

            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
            self.chat_history = [] # Optional: Can be used to maintain conversation history
                                   # For this simple chatbot, each run is stateless by default
                                   # unless history is explicitly passed to invoke.
        except Exception as e:
            print(f"Error initializing the Agent: {e}")
            raise

    def run(self, user_input: str) -> str:
        """
        Processes a user input and returns a chatbot response.

        Args:
            user_input: The user's message.

        Returns:
            The chatbot's response as a string.
        """
        try:
            # For a simple stateless chatbot, we just send the current user input.
            # If memory were required, self.chat_history would be updated and passed.
            messages = [HumanMessage(content=user_input)]
            
            response = self.llm.invoke(messages)
            
            if isinstance(response, AIMessage):
                return response.content
            else:
                return "I received an unexpected response format."
        except Exception as e:
            return f"An error occurred while processing your request: {e}"

if __name__ == "__main__":
    print("Initializing Chatbot Agent...")
    agent = None
    try:
        agent = Agent()
        print("Chatbot Agent initialized. Type 'exit' or 'quit' to end the conversation.")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Chatbot: Goodbye!")
                break
            
            response = agent.run(user_input)
            print(f"Chatbot: {response}")
            
    except ValueError as ve:
        print(f"Configuration Error: {ve}")
        print("Please ensure the GOOGLE_API_KEY environment variable is set.")
    except Exception as e:
        print(f"An unexpected error occurred during the main execution: {e}")