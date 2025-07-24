# Immune Generative AI

Este repositorio tiene como objetivo introducir los fundamentos de la **Inteligencia Artificial Generativa (GenAI)** y proporcionar una guía práctica para trabajar con **Modelos de Lenguaje de Gran Tamaño (LLMs)** usando herramientas de código abierto como **Ollama**.

---

## 📑 Índice de Contenido

- [🧠 ¿Qué es la Inteligencia Artificial Generativa?](#-qué-es-la-inteligencia-artificial-generativa)
- [📚 ¿Qué son los LLMs?](#-qué-son-los-llms)
- [🏢 Proveedores Comerciales de LLMs](#-proveedores-comerciales-de-llms)
- [🌍 Modelos de Código Abierto (actualizado 2025)](#-modelos-de-código-abierto-actualizado-2025)
- [🤗 Hugging Face](#-hugging-face)
- [🧰 Ollama](#-ollama)
- [🖥️ LM Studio](#-lm-studio)
- [🧩 AnythingLLM](#-anythingllm)
- [📚 Msty](#-msty)
- [🌐 Gradio](#-gradio)
- [🔗 Recursos adicionales](#-recursos-adicionales)

---

## 🧠 ¿Qué es la Inteligencia Artificial Generativa?

La **IA Generativa** es un campo de la inteligencia artificial que se centra en la creación de contenido nuevo y original a partir de datos existentes. Esto incluye texto, imágenes, código, música, video y más. Su auge reciente se debe al desarrollo de modelos de deep learning capaces de generar resultados sorprendentes en tareas creativas y cognitivas.

---

## 📚 ¿Qué son los LLMs?

Los **Large Language Models (LLMs)** son redes neuronales entrenadas con grandes cantidades de texto para aprender patrones del lenguaje humano. Estos modelos pueden:

- Generar texto coherente
- Resumir documentos
- Traducir entre idiomas
- Contestar preguntas
- Escribir código, entre muchas otras tareas.

---

## 🏢 Proveedores Comerciales de LLMs

A continuación se listan algunos de los principales proveedores que ofrecen LLMs accesibles mediante API o servicios cloud:

| Proveedor       | Modelo Principal            | Plataforma/API                   |
|------------------|-----------------------------|----------------------------------|
| **OpenAI**       | GPT-4o, GPT-4, GPT-3.5       | https://platform.openai.com     |
| **Anthropic**    | Claude 3                    | https://www.anthropic.com       |
| **Google**       | Gemini 1.5                  | https://ai.google.dev           |
| **Meta**         | LLaMA 4                     | https://ai.meta.com/llama       |
| **Alibaba**      | Qwen 2                      | https://qwenlm.github.io        |
| **DeepSeek AI**  | DeepSeek-V2, DeepSeek-Coder | https://deepseekcoder.github.io |
| **Mistral**      | Mistral/Mixtral             | https://mistral.ai               |
| **Cohere**       | Command-R+                  | https://cohere.com               |
| **Amazon**       | Titan                       | https://aws.amazon.com/bedrock  |

---

## 🌍 Modelos de Código Abierto (actualizado 2025)

A continuación se listan algunos de los LLMs open source más destacados y actualizados:

| Modelo           | Autor/Organización | Tamaños Disponibles | Versión Actual | Licencia      |
|------------------|--------------------|----------------------|----------------|----------------|
| **LLaMA**        | Meta               | 8B, 70B              | LLaMA 4        | Meta RAIL      |
| **Qwen**         | Alibaba            | 0.5B a 110B          | Qwen 2         | Apache 2.0     |
| **DeepSeek**     | DeepSeek AI        | 1.3B a 236B          | DeepSeek-V2    | MIT            |
| **Phi**          | Microsoft          | 3.8B, 7B             | Phi-3          | MIT            |
| **Gemma**        | Google             | 2B, 7B               | Gemma 1.1      | Apache 2.0     |
| **Mistral**      | Mistral AI         | 7B, Mixtral (12.7B MoE) | Mistral 7B / Mixtral 8x7B | Apache 2.0 |
| **Falcon**       | TII (UAE)          | 7B, 180B             | Falcon 180B    | Apache 2.0     |
| **Command-R**    | Cohere             | 35B                  | Command-R+     | RAIL           |

---

## 🤗 Hugging Face

[Hugging Face](https://huggingface.co/) es una de las plataformas más influyentes en el ecosistema de inteligencia artificial moderna. Su propósito es democratizar el acceso a modelos de IA, datasets y herramientas para investigación, desarrollo y despliegue de soluciones basadas en machine learning.

**¿Qué ofrece Hugging Face?**
- Un *hub centralizado* de modelos entrenados y datasets etiquetados.
- Librerías como `transformers`, `datasets`, `peft`, `diffusers` y `evaluate`.
- Espacios (*Spaces*) para ejecutar y compartir demos interactivas.
- La Hugging Face Hub API para integrar modelos en producción.
- Herramientas de fine-tuning, inferencia, cuantización y más.
- Una comunidad colaborativa activa de investigadores, empresas y desarrolladores.

**Usos comunes:**
- Integrar LLMs como LLaMA, Mistral o Phi en apps.
- Entrenar modelos personalizados.
- Evaluar y comparar arquitecturas de IA.
- Desplegar servicios de inferencia desde Hugging Face Inference Endpoints.
- Explorar nuevos modelos generativos, como Diffusion Models para imagen y audio.

---

## 🧰 Ollama

[**Ollama**](https://ollama.com) es una herramienta sencilla para ejecutar modelos LLM localmente, con soporte para múltiples modelos open-source preconfigurados.

### Requisitos

- macOS, Linux o Windows (con soporte WSL)
- Docker NO es necesario
- CPU moderna o GPU con soporte para aceleración (opcional)

### 📖 Documentación y APIs

- [Guía de la API de Ollama](../../wiki/ollama_api)  
- [SDK de Python para Ollama](../../wiki/ollama_python_sdk)
- [Creación y uso de Modelfile en Ollama](../../wiki/ollama_modelfile)

---

## 🖥️ LM Studio

[LM Studio](https://lmstudio.ai/) es una aplicación de escritorio que permite interactuar con modelos de lenguaje de manera local y visual. Funciona como una interfaz gráfica para Ollama y llama.cpp, y está pensada para usuarios no técnicos o que prefieren una experiencia similar a ChatGPT, pero con modelos locales.

**Características destacadas:**
- Descargar y ejecutar modelos directamente desde la app.
- Soporte para múltiples modelos open-source (LLaMA, Mistral, Phi, etc.).
- Personalización de temperatura, longitud de respuesta y formato.
- Compatible con macOS, Windows y Linux.

Ideal para experimentación, redacción de contenido, aprendizaje y evaluación rápida de modelos sin escribir código.

---

## 🧩 AnythingLLM

[AnythingLLM](https://anythingllm.com/) es una plataforma de código abierto que permite crear un **asistente privado impulsado por LLMs**, capaz de trabajar con tus propios datos, documentos y fuentes de conocimiento.  
[🐙 GitHub Repository](https://github.com/Mintplex-Labs/anything-llm)

**¿Qué puedes hacer con AnythingLLM?**
- Subir archivos (PDF, DOCX, TXT, etc.) y hacer preguntas sobre su contenido.
- Conectar fuentes externas como Notion, GitHub repos, sitios web y más.
- Utilizar diferentes backends como OpenAI, Ollama, Mistral, Hugging Face, entre otros.
- Desplegarlo en local, en servidores personales o en la nube.
- Administrar múltiples espacios y usuarios con control de acceso.

Una solución ideal para construir un **chat corporativo privado**, motores de búsqueda internos o asistentes personales de conocimiento con arquitectura plug-and-play.

---

## 📚 Msty

[Msty](https://msty.app/) es una herramienta especializada en la **visualización y análisis interno** de modelos de lenguaje.  
[🐙 GitHub Repository](https://github.com/linonetwo/msty)

Te permite explorar:
- **Tokens** generados en cada paso
- **Embeddings** vectoriales y distancias semánticas
- **Probabilidades de predicción** token a token

Ideal para investigadores, educadores y quienes quieren entender cómo "piensa" un modelo. Msty puede ayudarte a explicar errores, mejorar prompts o hacer debugging de outputs inesperados.

---

## 🌐 Gradio

[Gradio](https://www.gradio.app/) es una librería de Python que permite construir interfaces web para modelos de machine learning y deep learning en minutos. Ideal para prototipos, demos y validación con usuarios.

**Usos principales:**
- Crear formularios o chatbots con LLMs.
- Visualizar outputs de modelos de imagen, audio o NLP.
- Integrar con Hugging Face Spaces o notebooks.

Con unas pocas líneas de código puedes desplegar una interfaz intuitiva y compartible con cualquier persona.

---

## 🔗 Recursos adicionales

### 📚 Documentación y Comparativas
- [📖 Documentación oficial de Ollama](https://ollama.com/library) – Uso, modelos y configuración de Ollama.
- [📊 Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) – Comparativa actualizada de modelos open-source.
- [🧠 Awesome Open LLMs](https://github.com/Hannibal046/Awesome-LLMs) – Lista curada de modelos, datasets y herramientas.
- [🧪 Hugging Face Hub](https://huggingface.co/models) – Repositorio central de modelos preentrenados y datasets para IA generativa, NLP, visión y más.

### ⚙️ Frameworks y Librerías
- [📦 LlamaIndex](https://www.llamaindex.ai/) – Framework para crear aplicaciones de RAG (Retrieval-Augmented Generation).
- [🔁 LangGraph](https://www.langgraph.dev/) – Framework para flujos conversacionales multiestado con LLMs.
- [🧪 LangChain](https://www.langchain.com/) – Orquestación de agentes y flujos con LLMs, APIs y herramientas externas.
- [🤗 Transformers (Hugging Face)](https://huggingface.co/docs/transformers/index) – Librería para el uso de modelos de lenguaje en Python.

### 🛠️ Herramientas Locales y UI
- [🖥️ LM Studio](https://lmstudio.ai/) – Interfaz gráfica para descargar, ejecutar y chatear con LLMs localmente.
- [🧩 EverythingLLM](https://github.com/Evanz111/EverythingLLM) – Plataforma local y extensible para construir soluciones empresariales con LLMs.
- [🌐 Gradio](https://www.gradio.app/) – Crear interfaces web interactivas para modelos ML/LLM en minutos.
- [📚 Msty](https://github.com/linonetwo/msty) – Visualización y análisis de tokens, embeddings y procesos internos de los LLMs.
