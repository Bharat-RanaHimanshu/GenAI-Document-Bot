import streamlit as st
import sqlite3
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Here we use the Sqlite3 to store the user feedbacks and maintain it. It creates it if it doesn't exist already.
def init_db():
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (question TEXT, answer TEXT, status TEXT)''')
    conn.commit()
    conn.close()

def save_feedback(q, a, s):
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute("INSERT INTO feedback VALUES (?, ?, ?)", (q, a, s))
    conn.commit()
    conn.close()


# This takes care of losing the files when the user interacts as Streamlit reruns the script
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None  # The searchable Vector Store
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = [] # Tracking filenames to avoid duplicates
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0 # Used to reset the file uploader widget


# This ensure that the Embedding model and LLM are only loaded into GPU/RAM once, so that our app can run fast.
@st.cache_resource
def load_models():
    embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # # Check if key is in Streamlit Secrets (for Cloud) 
    # # or local environment (for testing)
    # groq_key = st.secrets["GROQ_API_KEY"] 
    
    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name="llama-3.3-70b-versatile", # Can also use : llama-3.1-8b-instant <-- Faster
        temperature=0
    )
    return embed, llm

embed_model, llm = load_models()
init_db()


# Here is the user interface
st.set_page_config(page_title="GenAI Document Bot", layout="wide")
st.title("GenAI Multi-Document Chatbot")

with st.sidebar:
    st.header("Document Library")

    # The 'Flash' button resets the entire app state.
    if st.button(" Flash/Clear All Documents", use_container_width=True):
        st.session_state.vector_db = None
        st.session_state.processed_files = []
        st.session_state.uploader_key += 1
        st.rerun()

    st.write("---")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type="pdf",
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )


# Here we check, if files are uploaded, we check if they are already in the 'Brain'. If not, we Load -> Split -> Embed -> Store.
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.processed_files:
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                # Save temp file to disk for the Loader to read
                file_path = uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # This loads PDF text
                loader = PyPDFLoader(file_path)
                data = loader.load()

                # Here we split text into 1000-character chunks with overlap which ensures no context is lost between chunks.
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_documents(data)

                # Here we create or ddd to Vector Database (FAISS)
                if st.session_state.vector_db is None:
                    st.session_state.vector_db = FAISS.from_documents(chunks, embed_model)
                else:
                    st.session_state.vector_db.add_documents(chunks)

                st.session_state.processed_files.append(uploaded_file.name)
                os.remove(file_path) # Cleanup

    st.success(f"Brain active with {len(st.session_state.processed_files)} files.")


# Here we do the Chat and Retrieval
if st.session_state.vector_db:
    user_query = st.text_input("Ask a question about your documents:")

    if user_query:
        # Similarity Search (Retrieve top 4 relevant chunks)
        retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 4})
        context_docs = retriever.invoke(user_query)

        # Augmentation (Merging context with the prompt)
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        template = """You are a policy expert. Use the following context to answer the question.
        If the answer is not in the context, say you don't know.

        Context: {context}

        Question: {question}

        Answer:"""

        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()

        # Generation (LLM creates the final response)
        with st.spinner("Analyzing across documents..."):
            response = chain.invoke({"context": context_text, "question": user_query})

        st.markdown(f"### Answer:\n{response}")

        # This is the CITATIONS SECTION
        # Here we show the exact file and page number.
        with st.expander(" View Citations (Source & Page)"):
            for i, doc in enumerate(context_docs):
                source_name = doc.metadata.get('source', 'Unknown')
                # Metadata page starts at 0, so we add 1 for the user.
                real_page_num = int(doc.metadata.get('page', 0)) + 1

                st.markdown(f"**Source {i+1}:** `{source_name}` — **Page {real_page_num}**")
                st.info(doc.page_content)

        # The FEEDBACK SYSTEM
        st.write("---")
        st.write("**Was this helpful?**")
        col1, col2 = st.columns([1, 10])
        if col1.button("👍"):
            save_feedback(user_query, response, "Helpful")
            st.toast("Feedback saved to database!")
        if col2.button("👎"):
            save_feedback(user_query, response, "Not Helpful")
            st.toast("Feedback saved for improvement.")
else:

    st.info("Upload one or multiple PDFs to begin.")


