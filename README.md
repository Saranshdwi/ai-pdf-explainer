# AI PDF Explainer

AI PDF Explainer is a Streamlit-based Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and interact with them using natural language questions.

The application extracts text from PDFs, generates embeddings using Sentence Transformers, stores them in a FAISS vector database, and retrieves relevant context to answer user queries using Groq's Llama models.

## Features

* Upload any PDF document
* Automatic text extraction and chunking
* Semantic search using FAISS
* Question-answering over document content
* Fast inference powered by Groq LLMs
* Interactive Streamlit interface
* Local embeddings (no embedding API cost)

## Tech Stack

* Streamlit
* LangChain
* FAISS
* Sentence Transformers
* Hugging Face Embeddings
* Groq (Llama 3.3 70B)
* PyPDF

## Project Structure

```text
ai-pdf-explainer/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml
```

## Installation

### Clone Repository

```bash
git clone https://github.com/Saranshdwi/ai-pdf-explainer.git
cd ai-pdf-explainer
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create `.streamlit/secrets.toml`

```toml
GROQ_API_KEY="your_groq_api_key"
```

## Run Application

```bash
streamlit run app.py
```

## How It Works

1. User uploads a PDF.
2. PDF text is extracted using PyPDFLoader.
3. Text is split into chunks.
4. Sentence Transformer embeddings are generated.
5. Chunks are stored in a FAISS vector database.
6. Relevant chunks are retrieved for each query.
7. Groq Llama model generates answers using retrieved context.
8. Response is displayed in the Streamlit interface.

## Future Improvements

* Chat history memory
* Multi-PDF support
* Source citations
* PDF summarization
* Conversational RAG
* Hybrid search
* Document metadata filtering

## Author

Saransh Dwivedi

Electrical Engineering Student | AI & Machine Learning Enthusiast

GitHub: https://github.com/Saranshdwi
LinkedIn: [www.linkedin.com/in/saransh-dwivedi](http://www.linkedin.com/in/saransh-dwivedi)
