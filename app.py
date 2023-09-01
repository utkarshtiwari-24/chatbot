'''import replicate

training = replicate.trainings.create(
  version="a16z-infra/llama-2-7b-chat:d24902e3fa9b698cc208b5e63136c4e26e828659a9f09827ca6ec5bb83014381",
  input={
    "train_data": "https://github.com/utkarshtiwari-24/chatbot/blob/a9cc69e114983c451b16d30c4ea72c350296322c/modelwise.jsonl",
  },
  destination="utkarshtiwari-24/chatbot"
)

print(training)

#r8_fHH9o5viKt7qhX35igYLAOgca3VVfyC16rvjr'''

import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

openai.api_key = st.secrets.openai_key
st.header("Modelwise Customer Support 💬 📚")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "I am Modelwise Chat Assistant, Chaze! How may I help you?"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="A potential customer has called in to inquire about our company Modelwise and its product Paitron and its capabilities. How would you handle the call as a trained and persuasive customer service agent? You need to try and convert the caller. "))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history


