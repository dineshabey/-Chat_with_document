import streamlit as st
import os
from PyPDF2 import PdfReader
import openpyxl
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

os.environ['GOOGLE_API_KEY'] = 'AIzaSyD8uzXToT4I2ABs7qo_XiuKh8-L2nuWCEM'

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_excel_text(excel_docs):
  text = ""
  for excel_doc in excel_docs:
      workbook = openpyxl.load_workbook(filename=excel_doc)
      for sheet in workbook:
          for row in sheet:
              for cell in row:
                  text += str(cell.value) + " "
  return text.strip()


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    llm = GooglePalm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

def get_user_input(user_question):
    with st.container():
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chatHistory = response['chat_history']
        file_contents = ""
        left , right = st.columns((2,1))
        with left:
            for i, message in enumerate(st.session_state.chatHistory):
                if i % 2 == 0:
                    st.write("User: ", message.content)
                else:
                    st.write("Bot: ", message.content)
            st.success("Done !")
        with right:
            for message in st.session_state.chatHistory:
                file_contents += f"{message.content}\n"
            file_name = "Chat_History.txt"

def main():
  st.set_page_config("DocChat")
  st.header("DocChat - Chat with multiple documents")
  st.write("---")
  with st.container():
    with st.sidebar:
      st.title("Settings")
      st.subheader("Upload Documents")
      st.markdown("**PDF files:**")
      pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
      if st.button("Process PDF file"):
        with st.spinner("Processing PDFs..."):
          raw_text = get_pdf_text(pdf_docs)
          text_chunks = get_text_chunks(raw_text)
          vector_store = get_vector_store(text_chunks)
          st.session_state.conversation = get_conversational_chain(vector_store)
          st.success("PDF processed successfully!")

      st.markdown("**Excel files:**")
      excel_docs = st.file_uploader("Upload Excel Files", accept_multiple_files=True)
      if st.button("Process Excel file"):
        with st.spinner("Processing Excel files..."):
          raw_text = get_excel_text(excel_docs)
          text_chunks = get_text_chunks(raw_text)
          vector_store = get_vector_store(text_chunks)
          st.session_state.conversation = get_conversational_chain(vector_store)
          st.success("Excel file processed successfully!")

  with st.container():
      st.subheader("Document Q&A")
      st.write('Ask a question : ')
      user_question = st.text_input("Ask a Question from the document")
      if "conversation" not in st.session_state:
          st.session_state.conversation = None
      if "chatHistory" not in st.session_state:
          st.session_state.chatHistory = None
      if user_question:
          get_user_input(user_question)

if __name__ == "__main__":
    main()
