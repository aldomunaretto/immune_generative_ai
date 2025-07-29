# Immune Generative AI

Este repositorio tiene como objetivo introducir los fundamentos de la **Inteligencia Artificial Generativa (GenAI)** y proporcionar una guía práctica para trabajar con **Modelos de Lenguaje de Gran Tamaño (LLMs)** usando herramientas de código abierto como **Ollama**.

---

## 📑 Índice de Contenido

- [🧠 ¿Qué es la Inteligencia Artificial Generativa?](#-qué-es-la-inteligencia-artificial-generativa)
- [📚 ¿Qué son los LLMs?](#-qué-son-los-llms)
- [🏢 Proveedores Comerciales de LLMs](#-proveedores-comerciales-de-llms)
- [🌍 Modelos de Código Abierto (actualizado 2025)](#-modelos-de-código-abierto-actualizado-2025)
- [📝 Context Engineering](#-context-engineering)
- [🎯 Fine-tuning](#-fine-tuning)
- [📂 Retrieval-Augmented Generation (RAG)](#-retrieval-augmented-generation-rag)
- [🤗 Hugging Face](#-hugging-face)
- [🧰 Ollama](#-ollama)
- [🖥️ LM Studio](#-lm-studio)
- [🧩 AnythingLLM](#-anythingllm)
- [📚 Msty](#-msty)
- [🌐 Gradio](#-gradio)
- [📊 Streamlit](#-streamlit)
- [🗣️ ElevenLabs](#️-elevenlabs)
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

## 📝 Context Engineering

El **Context Engineering** consiste en diseñar cuidadosamente los *prompts* y la información de entrada para optimizar las respuestas de un LLM sin necesidad de modificar sus pesos. 
Se centra en:
- Estructuración de *prompts* y *templates*.
- Inserción de *context windows* con ejemplos o instrucciones previas.
- Uso de técnicas como **Chain-of-Thought (CoT)**, **Few-Shot Prompting** y **ReAct**.

🔧 **Librerías comunes**:
- [DSPy](https://github.com/stanfordnlp/dspy) (Framework para context engineering basado en programas declarativos)
- [Guidance](https://github.com/microsoft/guidance)
- [Promptify](https://github.com/promptslab/Promptify)

---

## 🎯 Fine-tuning

El **Fine-tuning** es el proceso de ajustar los parámetros de un modelo previamente entrenado usando un conjunto de datos específico para una tarea concreta. 
Se utiliza para:
- Mejorar rendimiento en dominios especializados.
- Adaptar el estilo o formato de salida.
- Crear *instruction-tuned models* para casos concretos.

🔧 **Librerías y frameworks**:
- [Hugging Face Transformers](https://huggingface.co/transformers)
- [PEFT](https://github.com/huggingface/peft) (Parameter Efficient Fine-Tuning)
- [LoRA](https://github.com/microsoft/LoRA)
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio)
- [🦥 Unsloth](https://unsloth.ai/) (Framework optimizado para fine-tuning rápido y eficiente)
    - [🐙 GitHub Repository](https://github.com/unslothai/unsloth)
    - [🤗 Hugging Face Organization Card](https://huggingface.co/unsloth)
- [🦙 LLaMA-Factory](https://llamafactory.readthedocs.io/en/latest/)
    - [🐙 GitHub Repository](https://github.com/hiyouga/LLaMA-Factory)

---

## 📂 Retrieval-Augmented Generation (RAG)

El **RAG** combina la generación de texto con la recuperación de información externa en tiempo real. 
En lugar de confiar solo en el conocimiento interno del LLM, **recupera documentos relevantes** y los pasa como contexto al modelo antes de generar la respuesta.

🔧 **Librerías comunes**:
- [LangChain](https://www.langchain.com/)
- [LangGraph](https://www.langgraph.dev/) 
- [LlamaIndex](https://www.llamaindex.ai/)
- [Haystack](https://haystack.deepset.ai/)
- [Pinecone](https://www.pinecone.io/) (vector DB)
- [Weaviate](https://weaviate.io/) (vector DB)
- [Chroma](https://www.trychroma.com/) (vector DB)

**Casos de uso**:
- Chatbots empresariales con documentos privados.
- Búsqueda semántica combinada con LLMs.
- Sistemas de soporte y asistencia con información actualizada.

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

- [Repositorio de GitHub](https://github.com/ollama/ollama)
- [Documentación oficial de Ollama](https://github.com/ollama/ollama/tree/main/docss)
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

## 📊 Streamlit

[Streamlit](https://streamlit.io) es un framework en Python para crear **aplicaciones web interactivas de forma rápida**, ideal para prototipar interfaces con modelos de IA.

### 🔧 Características clave
- Interfaz muy sencilla basada en Python puro (sin necesidad de HTML/CSS/JS).
- Ideal para dashboards, demos de modelos y visualización de datos.
- Integración directa con librerías como `pandas`, `plotly`, `matplotlib` y APIs de LLMs.
- Permite desplegar aplicaciones fácilmente en la nube mediante [Streamlit Community Cloud](https://streamlit.io/cloud).

---

## 🗣️ ElevenLabs

[ElevenLabs](https://elevenlabs.io) es una plataforma líder en **generación de voz mediante IA**.  
Permite crear voces sintéticas realistas en múltiples idiomas y estilos, siendo ampliamente usada para aplicaciones de:

- **Narración de audiolibros y podcasts.**
- **Generación de diálogos en videojuegos.**
- **Conversión de texto a voz (TTS) en asistentes virtuales.**
- **Creación de personajes con voces personalizadas.**

### 🔧 Características clave
- Modelos de voz de alta fidelidad y expresividad.
- Soporte multilingüe y clonación de voz.
- API sencilla para integraciones en aplicaciones web y móviles.
- Planes de uso gratuito y de pago según volumen de caracteres.

### 📚 Librerías y SDKs
- [ElevenLabs Python SDK](https://pypi.org/project/elevenlabs/)
- API REST para integración con cualquier lenguaje.
- Plugins y conectores para aplicaciones de contenido multimedia.

---

## 🔗 Recursos adicionales

### 📚 Documentación y Comparativas

- [🤗 Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) – Comparativa actualizada de modelos open-source.
- [🧠 Awesome Open LLMs](https://github.com/Hannibal046/Awesome-LLMs) – Lista curada de modelos, datasets y herramientas.
- [🤗 Hugging Face Hub](https://huggingface.co/models) – Repositorio central de modelos preentrenados y datasets para IA generativa, NLP, visión y más.

### ⚙️ Frameworks y Librerías
- [🧪 LangChain](https://www.langchain.com/) – Orquestación de agentes y flujos con LLMs, APIs y herramientas externas.
- [🔁 LangGraph](https://www.langgraph.dev/) – Framework para flujos conversacionales multiestado con LLMs.
- [📦 LlamaIndex](https://www.llamaindex.ai/) – Framework para crear aplicaciones de RAG (Retrieval-Augmented Generation).
- [🤗 Transformers (Hugging Face)](https://huggingface.co/docs/transformers/index) – Librería para el uso de modelos de lenguaje en Python.

### Datasets para entrenamiento y ajuste fino (Fine-Tuning) de LLMs
- [The Pile](https://pile.eleuther.ai/)
- [HelpSteer: Helpfulness SteerLM Dataset](https://huggingface.co/datasets/nvidia/HelpSteer)
- [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
- [Anthropic_HH_Golden](https://huggingface.co/datasets/Unified-Language-Model-Alignment/Anthropic_HH_Golden)
- [Trelis Function Calling Dataset](https://huggingface.co/datasets/Trelis/function_calling_extended)
- [Dolma](https://huggingface.co/datasets/allenai/dolma)
- [Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)
- [Puffin](https://huggingface.co/datasets/LDJnr/Puffin)

### Personas de Interes
- [Maxime Labonne]()
    - [🐙 GitHub Repository](https://mlabonne.github.io/blog/)
    - [🤗 Hugging Face Page](https://huggingface.co/mlabonne)
    - [LLM Engineer's Handbook](https://github.com/PacktPublishing/LLM-Engineers-Handbook)
- [Colin Kealty (Bartowski)](https://x.com/bartowski1182)
    - [🐙 GitHub Repository](https://github.com/bartowski1182)
    - [🤗 Hugging Face Page](https://huggingface.co/bartowski)
    - [🤗 Hugging Face LM Studio Community Page](https://huggingface.co/lmstudio-community)