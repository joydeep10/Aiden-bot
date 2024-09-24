import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

# Check if API key is available
if not groq_api_key:
    st.error("GROQ API key not found. Please check your .env file.")
    st.stop()

try:
    myllm = ChatGroq(
        temperature=0.7,
        model="llama-3.1-70b-versatile",
        api_key=groq_api_key,
        streaming=True
    )
except Exception as e:
    st.error(f"Error initializing Groq models: {str(e)}")
    st.stop()

# Creating a prompt template for AntiDepression Bot
anti_depression_template = ChatPromptTemplate.from_messages([
    ("system", """Your name is Aidan, and you always refer to yourself as Aidan. You are a virtual assistant created to provide support, motivation, and practical solutions to people facing challenges. Aidan is compassionate, articulate, and deeply empathetic. You understand the nuances of human emotions and always strive to uplift and motivate. When responding to user questions, you must emulate Aidan's characteristics, providing thoughtful, caring, and constructive responses.
    Important Instructions:
    *Always respond with empathy and encouragement*
    *Provide practical advice whenever possible*"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{query}"),
    ("ai", "Aidan's response:")
])

# Create the chain using the new template
chain = anti_depression_template | myllm

# Create a dictionary to store chat histories
chat_histories = {}

# Function to get or create a chat history
def get_chat_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()  # Using ChatMessageHistory
    return chat_histories[session_id]

# Create RunnableWithMessageHistory
runnable_aiden = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="query",
    history_messages_key="history"
)


# Function to generate a response and update memory with streaming
def generate_response(query, session_id):
    response_placeholder = st.empty()
    full_response = ""

    for chunk in runnable_aiden.stream(
            {"query": query},
            config={"configurable": {"session_id": session_id}}
    ):
        full_response += chunk.content
        response_placeholder.markdown(full_response + "▌")
        time.sleep(0.02)

    response_placeholder.markdown(full_response)
    return full_response

# Streamlit UI setup
st.set_page_config(page_title="Aidan - A Joydeep Product", layout="wide")

# Create a custom header
st.markdown("""
# Aidan
##### ~ a Joydeep product
""", unsafe_allow_html=True)

# Short description of the chatbot's personality
st.write("""
Welcome to Aidan, your personal motivational assistant designed to uplift, support, 
and guide you through tough times. Aidan is here to listen, offer compassionate 
advice, and help you find practical solutions to your challenges. Let's talk!
""")

# Streamlit chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's going on in life......"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        session_id = st.session_state.get("session_id", "default")
        response = generate_response(prompt, session_id)

    st.session_state.messages.append({"role": "assistant", "content": response})