import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import huggingface_hub, google_palm
#import google.generativeai as palm

# Returns a string of text from a list of PDFs
# Uses PyPDF2 - PDF Reader
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs: #iterate over list of pdfs
        pdf_reader = PdfReader(pdf) #create pdf object (with pages)
        for page in pdf_reader.pages: #read each page, add it's text
            text += page.extract_text()
    return text

# Returns a list of text chunks from a single text string.
# Uses Langchain - langchain.text_splitter (CharacterTextSplitter)
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200, #two chunks could include the same portion of text, avoids losing meaning.
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Returns vector embeddings from text chunks.
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings() #AdaV2 - paid, but less resource exhausstive, not as good as instructor either.
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings) #FAISS - database for embeddings.
    return vectorstore

# Creates a conversation chain from a vector store
def get_conversation_chain(vectorstore):
    #llm = google_palm()
   # llm = ChatOpenAI(model_name="gpt-3.5-turbo") #model_name="text-davinci-003"
    llm = huggingface_hub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),  
        memory=memory,
        return_source_documents=True
    )
    return conversation_chain


def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.history = None
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

def handle_userinput(user_question): 
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    initialize_session_state()

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", type=["pdf"], accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):

                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore) 


if __name__ == '__main__':
    main()
