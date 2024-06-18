import streamlit as st

# set_page_config must be the first Streamlit command in the script
st.set_page_config(page_title="Notion Space Q&A Assistant", page_icon=":robot:")

import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate

# Set the OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_chain():
    """
    The `load_chain()` function initializes and configures a conversational retrieval chain for
    answering user questions.
    :return: The `load_chain()` function returns a ConversationalRetrievalChain object.
    """
    # Load OpenAI embedding model
    embeddings = OpenAIEmbeddings()
    
    # Load OpenAI chat model
    llm = ChatOpenAI(temperature=0)
    
    # Load our local FAISS index as a retriever
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Create memory 'chat_history' 
    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history")

    # Create the Conversational Chain
    chain = ConversationalRetrievalChain.from_llm(llm, 
                                                  retriever=retriever, 
                                                  memory=memory, 
                                                  get_chat_history=lambda h : h,
                                                  verbose=True)

    # Create system prompt
    template = """
    You are an AI assistant for answering questions about the Open OS Notion Space.
    You are given the following extracted parts of a long document and a question. Provide a conversational answer and link to specific parts of the notion space.
    If you don't know the answer, just say 'Sorry, I don't know... 😔'.
    Don't try to make up an answer.
    If the question is not about the Open OS Notion Space, politely inform them that you are tuned to only answer questions about the Blendle Employee Handbook.

    {context}
    Question: {question}
    Helpful Answer:"""

    # Add system prompt to chain
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)

    return chain

# Initialize and load the chain
chain = load_chain()

# Streamlit interface to interact with the chain
st.title("Notion Space Q&A Assistant")
st.write("Ask any question about the Open OS Notion Space.")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

with st.form(key='my_form', clear_on_submit=True):
    user_input = st.text_input("Your Question:", "")
    submit_button = st.form_submit_button(label='Ask')

if submit_button and user_input:
    # Append the user's message to the session state
    st.session_state['messages'].append({"role": "user", "content": user_input})
    
    # Get the response from the chain
    response = chain({"question": user_input})
    
    # Append the AI's response to the session state
    st.session_state['messages'].append({"role": "assistant", "content": response})

# Display the conversation
for message in st.session_state['messages']:
    if message["role"] == "user":
        st.write(f"**You:** {message['content']}")
    else:
        st.write(f"**Assistant:** {message['content']}")
