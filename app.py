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

def save_new_data(DB_FAISS_PATH):
    df = pd.read_csv('data.csv')
    new_data = {'pdf': [file_name],
            'db': [DB_FAISS_PATH]}
    new_df = pd.DataFrame(new_data)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv('data.csv', index=False)
    print("\nFile saved successfully!")


def ingest_into_vectordb(split_docs):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})
    db = FAISS.from_documents(split_docs, embeddings)
    vectorstore_dir = 'vectorstore'
    if not os.path.exists(vectorstore_dir):
        os.makedirs(vectorstore_dir)
    n_file_name= file_name.replace(".pdf", "")
    DB_FAISS_PATH = f'vectorstore/{n_file_name}'
    db.save_local(DB_FAISS_PATH)
    save_new_data(DB_FAISS_PATH)
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

def get_data():
    df = pd.read_csv('data.csv')
    key_value_dict = dict(zip(df['pdf'], df['db']))
    keys_list = list(key_value_dict.keys())
    return keys_list,key_value_dict



def load_from_vectordb(value):
    DB_FAISS_PATH = value
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(f"Could not find the VectorDB at {DB_FAISS_PATH}")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

def main():


    st.set_page_config(page_title="ManaGenie",
                    page_icon=":🤖:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("ManaGenie : Your PDF Multitasking Sidekick :🤖:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    pdf_list, values = get_data()
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        pdf_name = st.selectbox("Choose a preloaded vector file:", pdf_list)
        if st.button("Process"):
            with st.spinner("Processing"):
                if pdf_docs:
                    content, metadata = prepare_docs(pdf_docs)
                    split_docs = get_text_chunks(content, metadata)
                    vectorstore = ingest_into_vectordb(split_docs)
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)
                elif pdf_name:
                    value = values.get(pdf_name)
                    vectorstore = load_from_vectordb(value)
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)

if __name__ == '__main__':
    main()
