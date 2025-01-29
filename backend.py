import os
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
from langchain.chains import RetrievalQA
import nltk

nltk.download('averaged_perceptron_tagger')

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Add this to your .env file
PINECONE_INDEX_NAME = "medical"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

def is_valid_url(url):
    try:
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def extract_text_from_webpage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    extracted_text = "\n".join([para.get_text() for para in paragraphs])
    return extracted_text.strip()

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def store_embeddings(input_path):
    if input_path.startswith("http"):
        if not is_valid_url(input_path):
            return "❌ Error: URL is not accessible."
        
        if input_path.endswith(".pdf"):
            loader = OnlinePDFLoader(input_path)
            documents = loader.load()
            text_data = "\n".join([doc.page_content for doc in documents])
        else:
            text_data = extract_text_from_webpage(input_path)
            if not text_data:
                return "❌ Error: No readable text found on the webpage."
    else:
        documents = load_pdf(input_path)
        text_data = "\n".join([doc.page_content for doc in documents])
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(text_data)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    existing_indexes = [index.name for index in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine"
        )
    
    vectorstore = PineconeVectorStore.from_texts(
        texts=text_chunks,
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    return "✅ Data successfully processed and stored in Pinecone."

def query_chatbot(question, model="mixtral-8x7b-32768"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    try:
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings
        )
        retriever = docsearch.as_retriever()
    except Exception:
        retriever = None
    
    # Initialize Groq LLM
    llm = ChatGroq(
        temperature=0.7,
        groq_api_key=GROQ_API_KEY,
        model_name=model  # Using Mixtral 8x7B model
    )
    
    prompt_template = """
    You are a knowledgeable medical assistant. Use the provided context to answer the question accurately and professionally.
    If you don't know the answer or if the information isn't in the context, just say that you don't know.
    
    Context: {context}
    Question: {question}
    
    Helpful answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever if retriever else None,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain.run(question) if retriever else "No data source available. Please upload a document first."
