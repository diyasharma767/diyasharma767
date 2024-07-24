from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text
from unstructured.partition.docx import partition_docx

# Define a custom document class to hold content and metadata
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata else {}

class ChatBot:
    def __init__(self):
        load_dotenv()

        # Define the directory containing the files
        files_dir = "files/"
        file_paths = [os.path.join(files_dir, file) for file in os.listdir(files_dir)]

        # Load and extract text from the files
        docs = self.load_and_extract_text(file_paths)

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings()

        # Load or create FAISS index
        try:
            docsearch = FAISS.load_local("faiss_index", embeddings)
        except:
            docsearch = FAISS.from_documents(docs, embeddings)
            docsearch.save_local("faiss_index")

        # Define the repo ID and connect to Mixtral model on Huggingface
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0.3, "top_k": 50},
            huggingfacehub_api_token="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"
        )

        # Define the prompt template internally
        self.prompt_template = PromptTemplate(
            template="""
                Your task is to provide answers from the context provided
                Don't use external knowledge

                Context: {context}

                Question: {question}

                Answer:
            """,
            input_variables=['context', 'question']
        )

        # Set up RAG chain
        ret = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        self.rag_chain = (
            {"context": ret, "question": RunnablePassthrough()}
            | self.prompt_template
            | llm
            | StrOutputParser()
        )

    def load_and_extract_text(self, file_paths):
        docs = []
        for file_path in file_paths:
            if file_path.endswith(".pdf"):
                elements = partition_pdf(filename=file_path)
            elif file_path.endswith(".txt"):
                elements = partition_text(filename=file_path)
            elif file_path.endswith(".docx"):
                elements = partition_docx(filename=file_path)
            else:
                continue  # Skip unsupported file types

            # Convert elements to the required format
            for element in elements:
                docs.append(Document(page_content=str(element)))
        return docs

    def invoke(self, user_input):
        # Invoke RAG chain and remove prompt template from final output
        result = self.rag_chain.invoke(user_input)
        return result.strip()  # Remove extra newline added by prompt template

# Usage example
if __name__ == "__main__":
    bot = ChatBot()
    user_query = input("Ask me anything: ")
    result = bot.invoke(user_query)
    print(result)
