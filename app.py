import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import CTransformers
from htmlTemplate import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    print("Getting Texts...")
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    print("Splitting texts to chunks...")
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def load_llm():
    print("Loading llama2...")
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5,
    )
    print("LLama Loaded...")
    return llm


def get_vector_store(text_chunks):
    print("embedding and saving to vector stores...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print("Creating embeddings and saving to vectordb")
    vector_store.save_local("vectorstore/db_faiss")
    print("Done !!")


def create_conversation_change():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vector_store = FAISS.load_local("vectorstore/db_faiss", embeddings)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=load_llm(), retriever=vector_store.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_user_input(user_question):
    custom_prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
    # Combine the custom prompt and the user's question
    full_prompt = custom_prompt_template.format(context="", question=user_question)

    # Send the combined prompt to the LLM
    response = st.session_state.conversation({"question": full_prompt})

    # Extract the helpful answer from the LLM's response
    # This may require custom logic to parse the response and extract the answer

    # Display the helpful answer to the user
    st.write(response)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple pdfs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about yor documents")

    if user_question:
        handle_user_input(user_question)

    st.write(user_template.replace("{{MSG}}", "Hello Akhil"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "hello human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # Get text chunks
                text_chunks = get_text_chunks(raw_text)

                # Create Vector stores
                get_vector_store(text_chunks)

                # Create an instance of conversation change
                st.session_state.conversation = create_conversation_change()


if __name__ == "__main__":
    main()
