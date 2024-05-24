import streamlit as st
import sqlite3
from hashlib import sha256
from htmlTemplate import css, bot_template, user_template
import json
# Create a connection to SQLite database
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create table if not exist
c.execute('''CREATE TABLE IF NOT EXISTS users
             (username TEXT PRIMARY KEY, password TEXT, array_data TEXT)''')

conn.commit()



import streamlit as st
import os
from htmlTemplate import css, bot_template, user_template
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer, util
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pandas as pd
import time
file_name=None
llmtemplate = """[INST]
As an AI, provide accurate and relevant information based on the provided document. Your responses should adhere to the following guidelines:
- Answer the question based on the provided documents.
- Be direct and factual, limited to 50 words and 2-3 sentences. Begin your response without using introductory phrases like yes, no etc.
- Maintain an ethical and unbiased tone, avoiding harmful or offensive content.
- If the document does not contain relevant information, state "I cannot provide an answer based on the provided document."
- Avoid using confirmatory phrases like "Yes, you are correct" or any similar validation in your responses.
- Do not fabricate information or include questions in your responses.
- do not prompt to select answers. do not ask me questions
{question}
[/INST]
"""

def prepare_docs(pdf_docs):
    docs = []
    metadata = []
    content = []

    for pdf in pdf_docs:
        print(pdf.name)
        global file_name
        file_name=pdf.name
        pdf_reader = PyPDF2.PdfReader(pdf)
        for index, text in enumerate(pdf_reader.pages):
            doc_page = {'title': pdf.name + " page " + str(index + 1),
                        'content': pdf_reader.pages[index].extract_text()}
            docs.append(doc_page)
    for doc in docs:
        content.append(doc["content"])
        metadata.append({
            "title": doc["title"]
        })

    return content, metadata


def get_text_chunks(content, metadata):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=256,
    )
    split_docs = text_splitter.create_documents(content, metadatas=metadata)
    print(f"Split documents into {len(split_docs)} passages")
    return split_docs

def save_new_data(username):
    c.execute('''SELECT * FROM users WHERE username = ?''', (username,))
    user = c.fetchone()
    if user:
        username, password, array_json = user
        if array_json:
            array_data = json.loads(array_json)
        else:
            array_data = []
        print("Current Array Data:", array_data)
        array_data.append(file_name)
        c.execute('''UPDATE users SET array_data = ? WHERE username = ?''', (json.dumps(array_data), username))
        conn.commit()
        print("Array updated successfully!")
    else:
        print("User not found.")


def ingest_into_vectordb(split_docs,username):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})
    db = FAISS.from_documents(split_docs, embeddings)
    vectorstore_dir = 'vectorstore'
    if not os.path.exists(vectorstore_dir):
        os.makedirs(vectorstore_dir)
    n_file_name= file_name.replace(".pdf", "")
    DB_FAISS_PATH = f'vectorstore/{n_file_name}'
    db.save_local(DB_FAISS_PATH)
    save_new_data(username)
    return db




def get_conversation_chain(vectordb):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llama_llm = LlamaCpp(
    model_path="llama-2-7b-chat.Q4_K_M.gguf",
    n_gpu_layers=200,
    n_batch=512,
    temperature=0.75,
    max_tokens=200,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,
    n_ctx=3000)
    retriever = vectordb.as_retriever()
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(llmtemplate)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')
    conversation_chain = (ConversationalRetrievalChain.from_llm
                          (llm=llama_llm,
                           retriever=retriever,
                           #condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                           memory=memory,
                           return_source_documents=True))
    print("Conversational Chain created for the LLM using the vector store")
    return conversation_chain


def validate_answer_against_sources(response_answer, source_documents):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    similarity_threshold = 0.5  
    source_texts = [doc.page_content for doc in source_documents]

    answer_embedding = model.encode(response_answer, convert_to_tensor=True)
    source_embeddings = model.encode(source_texts, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(answer_embedding, source_embeddings)


    if any(score.item() > similarity_threshold for score in cosine_scores[0]):
        return True  

    return False  

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        print(i)
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            print(message.content)
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)




def load_from_vectordb(value):
    n_file_name= value.replace(".pdf", "")
    DB_FAISS_PATH = f'vectorstore/{n_file_name}'
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(f"Could not find the VectorDB at {DB_FAISS_PATH}")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db


def create_user(username, password):
    hashed_password = sha256(password.encode()).hexdigest()
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    # conn.commit()
    # array_data=['talal','bilal']
    # array_json = json.dumps(array_data)
    # hashed_password = sha256(password.encode()).hexdigest()
    # c.execute('''INSERT INTO users (username, password, array_data)
    #               VALUES (?, ?, ?)''', (username, hashed_password, array_json))
    conn.commit()

def verify_user(username, password):
    hashed_password = sha256(password.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_password))
    result = c.fetchone()
    return result is not None

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        print(i)
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            print(message.content)
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

import streamlit as st


def get_pdflist(username):
    c.execute('''SELECT * FROM users WHERE username = ?''', (username,))
    user = c.fetchone()
    if user:
        username, password, array_json = user
        array_data = json.loads(array_json) if array_json else []
        print("Username:", username)
        print("Password:", password)
        print("Array Data:", array_data)
        return array_data
    else:
        print("User not found.")


def chat_page(username):
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.markdown(
        f"""
        <style>
            .user-info {{
                position: fixed;
                top: 50px;
                right: 50px;
                padding: 5px;
                border-radius: 5px;
                transition: background-color 0.3s ease;
                cursor: pointer;
                font-size: 16px; 
                font-weight: bold; 
            }}
            .user-info:hover {{
                background-color: #e0e0e0;
            }}
        </style>
        <div class="user-info">{"🔓"}</i> {username}</div>
        """,
        unsafe_allow_html=True
    )


    
    st.header("ManaGenie : Your PDF Multitasking Sidekick🤖:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    pdf_list= get_pdflist(username)
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        pdf_name = st.selectbox("Choose a preloaded vector file:", pdf_list)
        if st.button("Process"):
            with st.spinner("Processing"):
                if pdf_docs:
                    # get pdf text
                    content, metadata = prepare_docs(pdf_docs)

                    # get the text chunks
                    split_docs = get_text_chunks(content, metadata)

                    # create vector store
                    vectorstore = ingest_into_vectordb(split_docs,username)
                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)
                elif pdf_name:
                    vectorstore = load_from_vectordb(pdf_name)
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)



def main_page():
    st.markdown(
        f"""
        <style>
            .user-info {{
                position: fixed;
                top: 50px;
                right: 320px;
                padding: 5px;
                border-radius: 5px;
                transition: background-color 0.3s ease;
                cursor: pointer;
                font-size: 30px; 
                font-weight: bold; 
            }}
            .user-info:hover {{
                background-color: #e0e0e0;
            }}
        </style>
        <div class="user-info">{"🔓"}</i>Welcome to ManaGenie : Your PDF Multitasking Sidekick🤖</div>
        """,
        unsafe_allow_html=True
    )
    animated_image_url = "https://mir-s3-cdn-cf.behance.net/project_modules/fs/200e8d139737079.6234b0487404d.gif" 
    st.image(animated_image_url, use_column_width=True)
    col1, col2, col3 = st.columns([20, 5, 20])
    st.markdown(
      f"""
      <style>
          .user-info1 {{
              position: fixed;
              top: 500px;
              right: 320px;
              padding: 5px;
              border-radius: 5px;
              transition: background-color 0.3s ease;
              cursor: pointer;
              font-size: 30px; 
              font-weight: bold; 
          }}
          .user-info:hover {{
              background-color: #e0e0e0;
          }}
      </style>
<div class="user-info1">Made By :
    <ul style="padding-left: 0;">
        <li style="margin-bottom: 10px; margin-left: 100px; font-weight: bold;">Haziq Ijaz</li>
        <li style="margin-bottom: 10px; margin-left: 100px;font-weight: bold;">Muhammad Talal</li>
        <li style="margin-bottom: 10px; margin-left: 100px;font-weight: bold;">Fatima Asim</li>
    </ul>
</div>
      """,
      unsafe_allow_html=True
  )



    with col2:
        if st.button("Login"):
            st.session_state.main_page1 = True
            st.empty()
            st.rerun()

def unlock_page():
    i = "lock.gif" 
    image_size = 500  

    col1, col2, col3 = st.columns([1, 2, 1]) 
    with col2:
        st.image(i, width=image_size)
        time.sleep(2.3)
        st.session_state.unlock = True
        st.rerun()
def main():
    if not st.session_state.get('main_page1', False):
        main_page()
        st.markdown(
            f"""
            <style>
                .user-info1 {{
                    position: fixed;
                    top: 500px;
                    right: 320px;
                    padding: 5px;
                    border-radius: 5px;
                    transition: background-color 0.3s ease;
                    cursor: pointer;
                    font-size: 30px; 
                    font-weight: bold; 
                }}
                .user-info:hover {{
                    background-color: #e0e0e0;
                }}
            </style>
      <div class="user-info1">Made By :
          <ul style="padding-left: 0;">
              <li style="margin-bottom: 10px; margin-left: 100px; font-weight: bold;">Haziq Ijaz</li>
              <li style="margin-bottom: 10px; margin-left: 100px;font-weight: bold;">Muhammad Talal</li>
              <li style="margin-bottom: 10px; margin-left: 100px;font-weight: bold;">Fatima Asim</li>
          </ul>
      </div>
            """,
            unsafe_allow_html=True
        )
    elif not st.session_state.get('unlock', False):
        st.empty()
        unlock_page()
    elif not st.session_state.get('logged_in', False):
        st.title("User Registration and Login")

        menu = ["Login", "Register"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Login":
            st.subheader("Login")
            username = st.text_input("Username")
            
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if verify_user(username, password):
                    st.success("Logged in as {}".format(username))
                    st.session_state.logged_in = True
                    st.session_state.user1 = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        elif choice == "Register":
            st.subheader("Create a New Account")
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            if new_password == confirm_password:
                if st.button("Register"):
                    create_user(new_username, new_password)
                    st.success("Account created successfully!")
            else:
                st.error("Passwords do not match")
    else:
        st.set_page_config(page_title="ManaGenie",
                        page_icon=":🤖:")
        st.write(css, unsafe_allow_html=True)

        chat_page(st.session_state.get('user1'))

if __name__ == '__main__':
    main()
