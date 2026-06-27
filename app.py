import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
# ---------------- PAGE CONFIG ----------------

st.set_page_config(page_title="PDF Explainer", page_icon="📄")

st.title("📄 PDF Explainer")
st.write("Upload a PDF and ask questions about its contents.")


# ---------------- PDF UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing PDF..."):

        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        prompt = PromptTemplate(
            template="""
You are a helpful assistant.

Answer ONLY from the provided context in about 300 words.

If the context is insufficient, say:
"I don't know based on the document."

Context:
{context}

Question:
{question}
""",
            input_variables=["context", "question"]
        )

        llm = ChatGroq(
            model="llama-3.3-70b-versatile"
        )

        def format_docs(retrieved_docs):
            return "\n\n".join(
                doc.page_content
                for doc in retrieved_docs
            )

        parallel_chain = RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )

        parser = StrOutputParser()

        chain = (
            parallel_chain
            | prompt
            | llm
            | parser
        )

    st.success("PDF processed successfully!")

    query = st.text_input(
        "Ask a question about the PDF"
    )

    if query:

        with st.spinner("Generating answer..."):
            answer = chain.invoke(query)

        st.subheader("Answer")
        st.write(answer)