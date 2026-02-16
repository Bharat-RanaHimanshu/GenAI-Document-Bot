# 🤖 GenAI Document Bot 📄

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://genai-document-bot-h4wkdyqr4z2xwlid4ffy7x.streamlit.app/)

A high-performance **Retrieval-Augmented Generation (RAG)** application that allows users to chat with their own PDF documents (Policies, SOPs, FAQs) securely and with full transparency.

---

## 🚀 Live Demo
**Click here to try it out:** [Live App on Streamlit Cloud](https://genai-document-bot-h4wkdyqr4z2xwlid4ffy7x.streamlit.app/)

---

## ✨ Key Features
* **Intelligent Retrieval:** Uses FAISS vector storage to find the most relevant document sections instantly.
* **Verifiable Citations:** Every answer includes the source document name and page number to prevent hallucinations.
* **Privacy-First:** Designed to work with local embeddings and cloud-scale inference via Groq.
* **User Feedback:** Integrated SQLite database to log queries and user ratings (Helpful/Not Helpful).

## 🛠️ Tech Stack
* **Framework:** [Streamlit](https://streamlit.io/)
* **Orchestration:** [LangChain](https://www.langchain.com/)
* **LLM:** [Groq](https://groq.com/) (llama-3.3-70b-versatile)
* **Embeddings:** [HuggingFace](https://huggingface.co/) (all-MiniLM-L6-v2)
* **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss)
* **Database:** SQLite

## 📖 How it Works
1.  **Upload:** User uploads one or multiple PDF documents.
2.  **Indexing:** Documents are split into chunks, embedded into vectors, and stored in a local FAISS index.
3.  **Querying:** When a question is asked, the system retrieves the top 4 most relevant chunks.
4.  **Generation:** The LLM generates a response based *only* on the provided context.

## ⚙️ Setup
1. Clone the repository: `git clone https://github.com/Bharat-RanaHimanshu/GenAI-Document-Bot.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your `.streamlit/secrets.toml` with your `GROQ_API_KEY`.
4. Run the app: `streamlit run app.py`
