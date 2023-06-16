from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from PIL import Image
import base64


def main():
    load_dotenv()

    # Add logo to header
    logo = "logo.png" 
    image = Image.open(logo)

    # Set page config
    st.set_page_config(page_title="LFLibrarian", layout="wide")

    # Add a header
    st.markdown("""
        <style>
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 36px;
                color: white;
                background: dark;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 3px 6px rgb(16,20,20), 0 3px 6px rgba(0,0,0,0.23);
                text-align: center;
                font-family: 'Roboto', sans-serif;
            }
            .logo {
                width: 120px;
                height: 120px;
            }
        </style>
        """, unsafe_allow_html=True)

    with open('logo.png', 'rb') as f:
        img = f.read()

    b64 = base64.b64encode(img).decode()

    st.markdown("""
        <div class='header'>
            <img class='logo' src='data:image/png;base64,{}' alt='Logo'>
            <span>FalkBot: The LFL Librarian</span>
        </div>
        """.format(b64), unsafe_allow_html=True)

    # Add file uploader in sidebar
    pdf = st.file_uploader("", type="pdf")

    # Add footer
    st.markdown("""
        <style>
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: dark;
                color: white;
                text-align: center;
                padding: 10px;
            }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("<div class='footer'>© 2023 Shanto. All rights reserved.</div>", unsafe_allow_html=True)

    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings
      embeddings = OpenAIEmbeddings()
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # show user input
      user_question = st.text_input("Ask a question to FalkBot:")
      if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
           
        st.write(response)
    

if __name__ == '__main__':
    main()
