import streamlit as st
import ollama
from typing import Dict, Generator
page_icon = r"data:image/png;base64,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"
import os

# ====== Create vector store and upload indexed data ======
@st.cache_resource
def download_models() -> None:
    try:
        os.system("curl -fsSL https://ollama.com/install.sh | sh")
        os.system(command = "ollama run phi3:instruct")
    except Exception as e:
        print(e)


download_models()

system_prompt = """Create flashcards based on this information. Flashcards must be in Q and A format. YOU ARE NEVER TO ADD ADDITIONAL INFORMATION OR KNOWLEDGE SOURCES BEYOND WHAT IS GIVEN. YOU MUST ONLY CREATE FLASHCARDS BASED ON THE INFORMATION PROVIDED. THE ONLY PERMITTED RESPONSES MUST BE IN THIS Q AND A FLASHCARD FORM.

Example:

My name is Dominic and I'm a traveling nurse. My favorite restaraunt is chipotle.

EXAMPLE RESPONSE: Q: What is the author's name? A: Dominic

Q: What is the author's occupation. 
A: Traveling Nurse.

Q: What is the author's favorite restaraunt?
A: Chipotle.

Now create flashcards based on the information below:

"""






def ollama_generator(model_name: str, messages: Dict) -> Generator:
    
    stream = ollama.chat(
        model=model_name, messages=messages, stream=True)
    for chunk in stream:
        yield chunk['message']['content']

st.title("Flashcard LLM")
st.text("")
st.text("")
if "selected_model" not in st.session_state:
    st.session_state.selected_model = ""
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
          {"role": "system", "content": "system_prompt", "avatar" : page_icon}
    ]

st.session_state.selected_model = st.selectbox(
    "Please select the model:", ["phi3:instruct"] )



# Display previous messages
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(name=message["role"], avatar=page_icon if message["role"]=="assistant" else None):
            st.markdown(message["content"])

if prompt := st.chat_input("Ask me a question!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    
    with st.chat_message(name="assistant", avatar = page_icon):
    # Generate and display AI's response
        resp_list = []
        response = st.write_stream(ollama_generator(
            st.session_state.selected_model, st.session_state.messages))

        response_str = "".join([resp  for resp in response])

        st.session_state.messages.append({"role": "assistant", "content": response_str})
