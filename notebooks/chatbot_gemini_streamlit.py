from google import genai
import streamlit as st
from dotenv import load_dotenv
import os

"""
Streamlit  es una biblioteca de Python que facilita la creación de aplicaciones web interactivas y visualizaciones de datos 
de manera rápida y sencilla. Permite a los desarrolladores construir interfaces de usuario atractivas y funcionales con pocas 
líneas de código, lo que la convierte en una herramienta popular para científicos de datos, analistas y desarrolladores que
desean compartir sus resultados y modelos de manera efectiva.

Este código crea una aplicación web simple que simula un chatbot similar a Gemini utilizando la API de Google.

para ejecutarlo debes usar el siguiente comando en la terminal:

streamlit run chatbot__gemini_streamlit.py

"""

load_dotenv()

st.title("Mi primer Chatbot con Gemini y Streamlit")

gemini_client = genai.Client()

if "gemini_model" not in st.session_state:
    st.session_state["gemini_model"] = "gemini-3-flash-preview"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hola, ¿?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        chat = gemini_client.chats.create(model=st.session_state["gemini_model"])

        history = []
        for m in st.session_state.messages[:-1]:
            role = "user" if m["role"] == "user" else "model"
            history.append({"role": role, "parts": [m["content"]]})

        stream = chat.send_message_stream(prompt)

        response = st.write_stream(chunk.text for chunk in stream)
    st.session_state.messages.append({"role": "assistant", "content": response})