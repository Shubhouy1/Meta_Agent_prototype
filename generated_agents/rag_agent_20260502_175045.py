import os
from typing import List

# LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

class Agent:
    """
    A RAG agent for answering questions from uploaded PDF documents using ChromaDB and Google Gemini.
    """
    def __init__(self):
        """
        Initializes the Agent with Google Gemini models and sets up ChromaDB for persistence.
        """
        try:
            # Initialize Google Gemini LLM and Embeddings
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            # Set up ChromaDB persistence directory
            self.persist_directory = "./chroma_db"
            os.makedirs(self.persist_directory, exist_ok=True) # Ensure directory exists

            # Initialize ChromaDB client. It will load from persist_directory if it exists,
            # or be ready to create a new one when documents are added.
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print(f"ChromaDB client initialized, using persistence directory: {self.persist_directory}")
            
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                add_start_index=True
            )
            print("Agent initialized successfully.")

        except Exception as e:
            print(f"Error during agent initialization: {e}")
            raise

    def upload_pdf(self, pdf_path: str):
        """
        Loads a PDF document, splits it into chunks, and adds it to the ChromaDB vector store.
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

            print(f"Loading PDF from {pdf_path}...")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} pages from PDF.")

            # Split documents into smaller chunks
            docs = self.text_splitter.split_documents(documents)
            print(f"Split into {len(docs)} chunks.")

            if not docs:
                print("No text chunks extracted from the PDF. Vector store will not be updated.")
                return

            # Add documents to ChromaDB
            # Chroma's add_documents method handles adding to an existing collection
            # or creating it if it's the first time documents are added to an empty client.
            self.vectorstore.add_documents(docs)
            
            # Persist the changes to disk
            self.vectorstore.persist()
            print(f"Successfully uploaded and processed {pdf_path}. Documents added to vector store.")

        except FileNotFoundError as e:
            print(f"Error uploading PDF: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during PDF upload: {e}")

    def run(self, user_input: str) -> str:
        """
        Answers a user question using RAG by retrieving relevant documents from ChromaDB
        and generating a response with Google Gemini.
        """
        try:
            # Check if the vectorstore actually contains documents.
            # ChromaDB's _collection.count() method can tell us this.
            if self.vectorstore._collection.count() == 0:
                return "Please upload a PDF document first using the 'upload_pdf' method before asking questions."

            print(f"Searching for relevant documents for question: '{user_input}'")
            # Retrieve relevant documents
            retrieved_docs: List[Document] = self.vectorstore.similarity_search(user_input, k=4)
            
            if not retrieved_docs:
                return "Could not find relevant information in the uploaded documents to answer your question."

            # Format the retrieved documents into a context string
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Construct the prompt for the LLM
            prompt = (
                "You are an AI assistant for question-answering over documents.\n"
                "Use the following retrieved context to answer the question.\n"
                "If you don't know the answer, just say that you don't know.\n"
                "Keep the answer concise and to the point.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {user_input}\n\n"
                "Answer:"
            )

            print("Generating response with LLM...")
            # Invoke the LLM
            response = self.llm.invoke(prompt)
            return response.content

        except Exception as e:
            print(f"An error occurred during the RAG process: {e}")
            return "An error occurred while processing your request. Please try again."

if __name__ == "__main__":
    # Example Usage
    agent = Agent()

    # Create a dummy PDF file for testing
    dummy_pdf_path = "example.pdf"
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter

        c = canvas.Canvas(dummy_pdf_path, pagesize=letter)
        c.drawString(100, 750, "This is a test document about Python programming.")
        c.drawString(100, 730, "Python is a high-level, interpreted programming language.")
        c.drawString(100, 710, "It is widely used for web development, data analysis, AI, and more.")
        c.drawString(100, 690, "LangChain is a framework for developing applications powered by language models.")
        c.drawString(100, 670, "ChromaDB is a vector database for building AI applications.")
        c.drawString(100, 650, "Google Gemini models offer powerful generative AI capabilities.")
        c.save()
        print(f"Created dummy PDF: {dummy_pdf_path}")
    except ImportError:
        print("reportlab not installed. Please install it (`pip install reportlab`) or provide a real PDF to test.")
        print("Skipping dummy PDF creation and upload.")
        dummy_pdf_path = None
    except Exception as e:
        print(f"Error creating dummy PDF: {e}")
        dummy_pdf_path = None

    if dummy_pdf_path:
        # Upload the dummy PDF
        agent.upload_pdf(dummy_pdf_path)

        # Ask questions
        print("\n--- Asking Questions ---")
        questions = [
            "What is Python?",
            "What is LangChain used for?",
            "What is ChromaDB?",
            "What are Google Gemini models?",
            "Who created Python?", # This answer is not in the document
            "Tell me about interpreted programming languages."
        ]

        for i, question in enumerate(questions):
            print(f"\nQuestion {i+1}: {question}")
            answer = agent.run(question)
            print(f"Answer: {answer}")
    else:
        print("\nSkipping QA session as no PDF was uploaded.")
        print("To test, manually create a 'example.pdf' or install 'reportlab' and run again.")

    # The chroma_db directory will persist as per the problem constraints (no os.remove()).