from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import numpy as np

def main():
    load_dotenv()
    st.set_page_config(page_title="The Digital Tolkien")
    st.header("Ask your question to J.R.R Tolkien 💬")
    
    # List of .txt files with content
    txt_files = [
        "LOTR.txt",
        "FALL_OF_GONDOLIN.txt",
        "CHILDREN_OF_HURIN.txt",
        "SILMARILLION.txt",
        "UNFINISHED_TALES.txt",
        "LETTERS.txt",
    ]
    
    # Read content from .txt files and combine into one large text
    text = ""
    for file in txt_files:
        with open(file, "r", encoding="utf-8") as f:
            text += f.read()

    # Split the text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create knowledge base with embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # User input
    user_question = st.text_input("Ask your question here:")
    if user_question:
        docs = knowledge_base.similarity_search(user_question,k=3)
        
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            print(cb)

        st.write(response)

if __name__ == '__main__':
    main()
