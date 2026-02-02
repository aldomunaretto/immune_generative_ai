from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
import os

"""
Streamlit  es una biblioteca de Python que facilita la creación de aplicaciones web interactivas y visualizaciones de datos 
de manera rápida y sencilla. Permite a los desarrolladores construir interfaces de usuario atractivas y funcionales con pocas 
líneas de código, lo que la convierte en una herramienta popular para científicos de datos, analistas y desarrolladores que
desean compartir sus resultados y modelos de manera efectiva.

Este código crea una aplicación web simple que simula un chatbot similar a ChatGPT utilizando la API de OpenAI.

para ejecutarlo debes usar el siguiente comando en la terminal:

streamlit run chatbot_openai_streamlit.py

"""

load_dotenv()

st.title("Quiero ser ChatGPT")

client = OpenAI()

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-5-nano"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})