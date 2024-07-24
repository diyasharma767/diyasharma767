from flask import Flask, request, render_template
import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from unstructured.partition.pdf import partition_pdf

app = Flask(__name__)
# Define a custom document class to hold content and metadata
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata else {}

class ChatBot:
    def __init__(self):
        load_dotenv()

        pdf_path = "PYTHON.pdf"
        # Load and extract text from the PDF
        elements = partition_pdf(filename=pdf_path)

        # Convert elements to the required format
        docs = [Document(page_content=str(element)) for element in elements]

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
            Your task is to provide answers from the context provided\n
            Don't use external knowledge\n

            Context:{context}

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

    def invoke(self, user_input):
        # Invoke RAG chain and remove prompt template from final output
        result = self.rag_chain.invoke(user_input)
        return result.strip()  # Remove extra newline added by prompt template

# Instantiate the ChatBot
chatbot = ChatBot()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_query = request.form["question"]
        result = chatbot.invoke(user_query)
        return render_template("index.html", question=user_query, answer=result)
    return render_template("index.html", question="", answer="")

if __name__ == "__main__":
    app.run(debug=True)
