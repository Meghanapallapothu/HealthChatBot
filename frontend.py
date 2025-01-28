import os
import re
import tempfile
import streamlit as st
import nltk
from dotenv import load_dotenv
from backend import store_embeddings, query_chatbot

# Ensure nltk is ready
try:
    nltk.data.find('corpora/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "medical"

# Streamlit page setup
st.set_page_config(page_title="Medical-Bot", layout="wide")
st.title("🩺 Medical-Bot - AI-powered Medical Assistant")

# Sidebar - Data Source Selection
st.sidebar.header("Data Source Selection")
data_source = st.sidebar.radio("Choose data source:", ["Upload a PDF", "Enter a URL", "Use Default Data"])

pdf_file = None
url_input = ""

def is_valid_url(url):
    pattern = re.compile(r"^(https?://)?(www\.)?[\w-]+(\.[\w-]+)+[/#?]?.*$")
    return re.match(pattern, url)

if data_source == "Upload a PDF":
    pdf_file = st.sidebar.file_uploader("📂 Upload a PDF file", type=["pdf"])
elif data_source == "Enter a URL":
    url_input = st.sidebar.text_input("🔗 Paste a URL:")
elif data_source == "Use Default Data":
    st.sidebar.write("📚 Using preloaded medical data.")

# Initialize session state for chat history
if "queries" not in st.session_state:
    st.session_state.queries = []

# Process Data Button with Loading Indicator
if st.sidebar.button("⚡ Process Data"):
    with st.spinner("📄 Processing data... Please wait!"):
        try:
            if pdf_file:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(pdf_file.read())
                    result = store_embeddings(tmp_file.name)
                    st.sidebar.success("✅ PDF processed successfully!")
            elif url_input:
                if is_valid_url(url_input):
                    result = store_embeddings(url_input)
                    st.sidebar.success("✅ URL processed successfully!")
                else:
                    st.sidebar.error("❌ Invalid URL! Please enter a valid URL.")
            elif data_source == "Use Default Data":
                result = store_embeddings("Medical_book.pdf")
                st.sidebar.success("✅ Default data processed!")
        except Exception as e:
            st.sidebar.error(f"❌ Error processing data: {str(e)}")

# Ask a medical question
st.header("🤖 Ask Your Medical Question")
question = st.text_area("Type your query below:")

if st.button("💬 Submit Query") and question:
    with st.spinner("🤔 Generating response... Please wait!"):
        try:
            response = query_chatbot(question)
            st.session_state.queries.append((question, response))
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")

# Display previous queries in a Chat-style UI
st.subheader("💬 Chat History")
for idx, (q, r) in enumerate(st.session_state.queries):
    with st.chat_message("user"):
        st.write(f"**You:** {q}")
    with st.chat_message("assistant"):
        st.write(f"**Bot:** {r}")

# Footer
st.markdown("---")
st.markdown("🔍 **Medical-Bot** - AI-powered assistant for medical information. 💡")