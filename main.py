from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv
from unstructured.partition.auto import partition

# Define a custom document class to hold content and metadata
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata else {}

class ChatBot:
    def __init__(self):
      load_dotenv()

      # Load documents using unstructured
      raw_documents = partition("PYTHON.txt")

      # Convert documents to the required format
      docs = [Document(doc.text) for doc in raw_documents]

      # Split documents
      text_splitter = CharacterTextSplitter(chunk_size=70, chunk_overlap=10)
      split_docs = text_splitter.split_documents(docs)

      # Initialize embeddings
      embeddings = HuggingFaceEmbeddings()

      # Load or create FAISS index
      try:
          docsearch = FAISS.load_local("faiss_index", embeddings)
      except:
          docsearch = FAISS.from_documents(split_docs, embeddings)
          docsearch.save_local("faiss_index")

      # Define the repo ID and connect to Mixtral model on Huggingface
      repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
      llm = HuggingFaceHub(
            repo_id=repo_id, 
            model_kwargs={"temperature": 0.3, "top_k": 50}, 
            huggingfacehub_api_token="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"
        )

      # Define the prompt template
      template = """
        Question: {question}
        Answer:
        """

      prompt = PromptTemplate(
            template=template,
            input_variables=['context', 'question']
        )
      ret=docsearch.as_retriever(search_type="similarity",
        search_kwargs={
         "k": 10  # Number of top results to return
        })

      # Define the RAG chain
      self.rag_chain = (
          {"context": ret, "question": RunnablePassthrough()} 
            | prompt 
            | llm 
            | StrOutputParser()
        )

    def invoke(self, user_input):
        return self.rag_chain.invoke(user_input)

# Usage example
if __name__ == "__main__":
    bot = ChatBot()
    user_query = input("Ask me anything: ")
    result = bot.invoke(user_query)
    print(result)






