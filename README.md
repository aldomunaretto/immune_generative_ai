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
- [🧪 Evaluación y Métricas](#-evaluación-y-métricas)
- [🔐 Seguridad, Alineación y Riesgos](#-seguridad-alineación-y-riesgos)
- [💰 Optimización de Costos](#-optimización-de-costos)
- [🛰️ Observabilidad y Trazabilidad](#-observabilidad-y-trazabilidad)
- [🧪 Testing de Prompts](#-testing-de-prompts)
- [🧵 Gestión de Contexto Extendido](#-gestión-de-contexto-extendido)
- [🛣️ Roadmap de Aprendizaje Sugerido](#️-roadmap-de-aprendizaje-sugerido)
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

| Proveedor       | Modelos Destacados (2026)                              | Plataforma/API                   | Notas |
|-----------------|--------------------------------------------------------|----------------------------------|-------|
| **OpenAI**      | GPT-5.2, o3, o1-pro, GPT-4o                            | https://platform.openai.com      | Soporte multimodal, Realtime API |
| **Anthropic**   | Claude 4.5 Opus (Thinking), Claude 4 Sonnet            | https://www.anthropic.com        | Foco en seguridad. Excelente en matices lingüísticos y razonamiento complejo con modo "Thinking". |
| **Google**      | Gemini 3 Pro / Nano Banana / Veo3                      | https://ai.google.dev            | Contexto largo, multimodal nativo y foco en razonamiento + agentes. |
| **Meta**        | Llama 4 Scout, Llama 4 Maverick (pesos + API)          | https://ai.meta.com/llama        | “Open-weight” (con condiciones de licencia), opción de self-hosting + acceso vía API.|
| **Alibaba**     | Qwen3 (familia), Qwen3-Max, (línea Qwen3-Next)         | https://qwenlm.github.io         | Mucho empuje en modelos “hybrid reasoning” y escalado; Max aparece como tope de gama.|
| **DeepSeek AI** | DeepSeek-V3.2, DeepSeek-R1                             | https://deepseekcoder.github.io  | Enfocado en razonamiento/código con releases frecuentes (V3.x) y línea R1. Especializado en código. |
| **Mistral**     | Mistral Large 3, Mistral 3 (14B/8B/3B)                 | https://mistral.ai               | Oferta muy sólida en open models + opciones enterprise; nueva generación “Mistral 3”.|
| **Cohere**      | Command A, Command R / R+.                             | https://cohere.com               | Orientado a enterprise + RAG + tool use; Cohere recomienda Command A como “latest” frente a R+. |
| **AWS (Bedrock)**      | Amazon Nova (Premier/Pro/Lite/Micro), + Titan (embeddings / image / legacy text)   | https://aws.amazon.com/bedrock   | Nova es la línea moderna multimodal/agentic; Titan sigue presente (especialmente embeddings/imágenes) y parte de text se migra hacia Nova.|
| **xAI** | Grok 4, Grok 3 | https://docs.x.ai/ | API con modelos versionados/aliases; Grok 4 aparece como oferta actual. |
| **AI21 Labs** | Jamba, Jamba2| https://docs.ai21.com/ | Modelos basados en la línea Claude, con enfoque en NLP avanzado. |

### 🚀 Tendencias Clave de 2026
- Modelos de Razonamiento (Reasoning): Ya no solo predicen la siguiente palabra; modelos como o3 o Claude 4.5 Thinking utilizan "cadena de pensamiento" interna antes de responder, lo que reduce drásticamente las alucinaciones en tareas lógicas.
- Agentes Nativos: La mayoría de los modelos actuales (especialmente Llama 4 y GPT-5.2) están diseñados para usar herramientas de forma autónoma, permitiendo crear flujos de trabajo sin intervención humana constante.
- Eficiencia Extrema: Los modelos "Small" o "Flash" de 2026 son hoy más potentes de lo que era GPT-4 en su lanzamiento, permitiendo inferencia local en dispositivos con gran precisión.

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

## 📝 Prompt Engineering

La ingeniería de prompts (Prompt Engineering) es el arte y la ciencia de diseñar entradas (prompts) para LLMs con el fin de obtener salidas más precisas, seguras y previsibles sin alterar los pesos del modelo. Combina diseño de instrucciones, estructuración del contexto, selección de ejemplos y restricciones de formato para alinear la respuesta con requisitos funcionales y de negocio.

Técnicas comunes
- Mensajes de sistema/rol: separar instrucciones de alto nivel (system) de la petición del usuario.
- Zero‑shot vs Few‑shot: proporcionar 0 o varios ejemplos para guiar estilo y formato.
- Prompt templates: plantillas parametrizables y versionadas.
- Constraints estructurales: schema JSON, regex o formatos esperados para validar salidas.
- Temperature / sampling: controlar creatividad vs determinismo.
- Chain-of-Thought (CoT) y pasos intermedios: usar pasos explicativos controlados para tareas de razonamiento (con precaución).
- Tool‑use / function calling: definir llamadas a herramientas y formatos de intercambio.
- Robustness: testear contra prompt injection y entradas adversarias.

Ejemplos prácticos

- Zero‑shot (instrucción clara)
```
System: Eres un redactor técnico conciso.
User: Resume el siguiente texto en 3 bullets con lenguaje para ejecutivos.
[DOCUMENTO...]
```

- Few‑shot (estilo y formato)
```
User: Ejemplo 1:
Input: ¿Qué es X?
Output: X es... (1-2 frases)

User: Ejemplo 2:
Input: Cómo configurar Y?
Output: 1) Paso A 2) Paso B

User: Ahora haz lo mismo para: [nueva pregunta]
```

- Output estructurado (JSON schema)
```
System: Responde SOLO en JSON con campos: { "summary": string, "impact": "low|medium|high", "actions": [string] }
User: Resume este informe y sugiere acciones.
[INFORME...]
```

- Role play / persona
```
System: Eres un analista de seguridad con 10 años de experiencia.
User: Identifica 3 riesgos clave y una mitigación por cada uno.
```

Buenas prácticas breves
- Definir rol, objetivo, audiencia y formato (ROCE).
- Proveer ejemplos representativos y contraejemplos.
- Validar y testear con mutaciones adversarias.
- Versionar templates y registrar prompt final utilizado en producción.
- Preferir restricciones estructurales (JSON schema) en lugar de solo instrucciones abiertas.

Lecturas y herramientas
- [Prompt Engineering Guide](https://promptingguide.ai/)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models — Wei et al.](https://arxiv.org/pdf/2201.11903)
- [Rebuff: Detecting Prompt Injection Attacks](https://www.blog.langchain.com/rebuff/)
- [Guidance](https://github.com/guidance-ai/guidance)
- [Promptfoo: LLM evals & red teaming](https://github.com/promptfoo/promptfoo)

Referencias rápidas
- Guidance, Promptfoo, Promptify — utilidades para crear, testar y versionar prompts.


### ROCE — Framework para diseñar prompts

ROCE es un acrónimo que resume cuatro elementos clave para construir prompts claros y efectivos al interactuar con LLMs:

- R — Rol: indica la perspectiva o identidad del modelo (p. ej. "Eres un analista de seguridad senior").
- O — Objetivo: define la meta concreta del prompt (p. ej. "Resumir los riesgos en 3 puntos accionables").
- C — Contexto: aporta datos relevantes, restricciones y cualquier información necesaria (documentos, formato, audiencia).
- E — Ejemplo: muestra ejemplos de entrada/salida o el formato deseado para guiar la respuesta.

Por qué usar ROCE:
- Aumenta la precisión y relevancia de las respuestas.
- Reduce ambigüedad y alucinaciones.
- Facilita respuestas con formato predecible y verificable.

Plantilla rápida:

```
# ROLE
Eres un [profesión/experto] con [N años] de experiencia en [área].
Tu especialidad es [skill específico].

# OBJECTIVE
Tu tarea es [acción específica] que cumpla:
- [Criterio 1 de éxito]
- [Criterio 2 de éxito]
- [Criterio 3 de éxito]

# CONTEXT
- Audiencia: [descripción detallada]
- Formato: [estructura específica]
- Tono: [estilo de comunicación]
- Restricciones: [límites claros]
- NO hacer: [prohibiciones explícitas]

# EXAMPLE
[Muestra concreta del output deseado]
```

Consejos prácticos:
- Sé explícito y conciso en cada elemento.
- Proporciona ejemplos representativos.
- Itera: refina rol, contexto o ejemplos si la salida no coincide con lo esperado.
- Combínalo con técnicas como few-shot, instrucciones de sistema y constraints (JSON schema, límites de tokens) para mayor robustez.

---

## 📝 Context Engineering

El **Context Engineering** consiste en diseñar cuidadosamente los *prompts* y la información de entrada para optimizar las respuestas de un LLM sin necesidad de modificar sus pesos. 
Se centra en:
- Estructuración de *prompts* y *templates*.
- Inserción de *context windows* con ejemplos o instrucciones previas.
- Uso de técnicas como **Chain-of-Thought (CoT)**, **Few-Shot Prompting** y **ReAct**.

📖 **Lecturas sobre Context Engineering**:
- [A Survey of Context Engineering for Large Language Models](https://arxiv.org/abs/2507.13334)
- [Context Rot: How Increasing Input Tokens Impacts LLM Performance](https://research.trychroma.com/context-rot)
- [How Long Contexts Fail](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html)
- [Why “Context Engineering” Matters](https://www.dbreunig.com/2025/07/24/why-the-term-context-engineering-matters.html)
- [Context Engineering for Agents](https://blog.langchain.com/context-engineering-for-agents/)
- [Optimizing LangChain AI Agents with Contextual Engineering](https://levelup.gitconnected.com/optimizing-langchain-ai-agents-with-contextual-engineering-0914d84601f3)
- [Prompt Injection Exploits](https://blog.langchain.dev) (riesgos)
- [Evaluation Harness for Prompts](https://github.com) (buscar frameworks)

🔧 **Librerías comunes**:
- [DSPy](https://dspy.ai/) (Framework para context engineering basado en programas declarativos)
    - [🐙 GitHub Repository](https://github.com/stanfordnlp/dspy) 
- [Guidance](https://github.com/microsoft/guidance)
- [Promptify](https://github.com/promptslab/Promptify)
- [Promptfoo (testing)](https://github.com/promptfoo/promptfoo)

---

## 🎯 Fine-tuning

El **Fine-tuning** es el proceso de ajustar los parámetros de un modelo previamente entrenado usando un conjunto de datos específico para una tarea concreta. 
Se utiliza para:
- Mejorar rendimiento en dominios especializados.
- Adaptar el estilo o formato de salida.
- Crear *instruction-tuned models* para casos concretos.

### 🔧 Librerías y frameworks:
- [Hugging Face Transformers](https://huggingface.co/transformers)
- [PEFT](https://github.com/huggingface/peft) (Parameter Efficient Fine-Tuning)
- [LoRA](https://github.com/microsoft/LoRA)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio)
- [🦥 Unsloth](https://unsloth.ai/) (Framework optimizado para fine-tuning rápido y eficiente)
    - [🐙 GitHub Repository](https://github.com/unslothai/unsloth)
    - [🤗 Hugging Face Organization Card](https://huggingface.co/unsloth)
- [🦙 LLaMA-Factory](https://llamafactory.readthedocs.io/en/latest/)
    - [🐙 GitHub Repository](https://github.com/hiyouga/LLaMA-Factory)

### 🪢 Datasets para entrenamiento y ajuste fino (Fine-Tuning) de LLMs:
- [The Pile](https://pile.eleuther.ai/)
- [HelpSteer: Helpfulness SteerLM Dataset](https://huggingface.co/datasets/nvidia/HelpSteer)
- [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
- [Anthropic_HH_Golden](https://huggingface.co/datasets/Unified-Language-Model-Alignment/Anthropic_HH_Golden)
- [Trelis Function Calling Dataset](https://huggingface.co/datasets/Trelis/function_calling_extended)
- [Dolma](https://huggingface.co/datasets/allenai/dolma)
- [Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)
- [Puffin](https://huggingface.co/datasets/LDJnr/Puffin)
- [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
  - [🐙 GitHub Repository](https://github.com/tatsu-lab/stanford_alpaca)

### 📄 Formatos para Fine Tuning: Alpaca vs ShareGPT

Los formatos **Alpaca** y **ShareGPT** son los más utilizados para el fine‑tuning supervisado de LLMs, especialmente con frameworks como **LLaMA‑Factory** y **Unsloth**.

#### 🧪 Formato Alpaca

📂 Estructura típica (JSON)

```json
{
  "instruction": "...",    
  "input": "...",          
  "output": "...",         
  "system": "...",         
  "history": [             
    ["instrucción anterior", "respuesta anterior"]
  ]
}
```

- `instruction`: Pregunta o instrucción del humano (requerido).
- `input`: Contexto adicional (opcional).
- `output`: Respuesta esperada del modelo (requerido).
- `system`: Mensaje del sistema (opcional).
- `history`: Rondas anteriores en conversaciones multironda (opcional).

##### ✅ Ventajas
- Muy simple y ampliamente adoptado.
- Ideal para datasets de instrucción-respuesta de una sola ronda.

##### ⚠️ Limitaciones
- No tiene estructura nativa para roles múltiples o multironda avanzada.
- Se apoya en tokens de separación (`###` o EOS).


#### 🧵 ShareGPT

##### 📂 Estructura típica (JSON)

```json
{
  "conversations": [
    { "from": "human", "value": "…instrucción humana…" },
    { "from": "gpt",   "value": "…respuesta del modelo…" },
    { "from": "function_call", "value": "…argumentos de herramienta…" },
    { "from": "observation",   "value": "…resultado de herramienta…" }
  ],
  "system": "...",
  "tools": "..."
}
```

- Permite múltiples roles: `human`, `gpt`, `function_call`, `observation`.
- Diseñado para conversaciones multironda y llamadas a funciones.

##### ✅ Ventajas
- Soporta roles y contextos complejos.
- Ideal para datasets de diálogo real y herramientas.

##### ⚠️ Limitaciones
- Más complejo de preparar que Alpaca Format.


#### 📦 Soporte en Frameworks

- **LLaMA‑Factory**: Soporta ambos formatos mediante configuración en `dataset_info.json`.
- **Unsloth**:
  - Permite convertir con `standardize_sharegpt()`.
  - Ofrece `conversation_extension` para simular multironda desde Alpaca.


#### 📊 Comparativa Rápida

| Aspecto                  | Alpaca Format                         | ShareGPT Format                              |
|--------------------------|----------------------------------------|----------------------------------------------|
| Instrucción + respuesta  | ✅                                    | ✅                                           |
| Multi-turno              | ⚠️ Opcional, sin estructura fija       | ✅ Nativo                                    |
| Roles adicionales        | ❌                                     | ✅ function_call, observation, etc.          |
| Complejidad              | 🔹 Baja                               | 🔹 Media-Alta                                |
| Conversión disponible    | 🔸 Limitada                            | ✅ Via Unsloth                               |


#### 🧠 ¿Cuál elegir?

- **Alpaca** → Para datasets simples de instrucciones/respuestas.
- **ShareGPT** → Para diálogos multironda, roles múltiples y escenarios con herramientas.

#### 📄 Lecturas:

- [How to create a custom Alpaca instruction dataset for fine-tuning LLMs](https://zackproser.com/blog/how-to-create-a-custom-alpaca-dataset)

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

**Estrategias avanzadas:**
- Hybrid Search (BM25 + vector)
- Re-rankers (Cross-Encoder, ColBERT)
- Chunking adaptativo (basado en densidad semántica)
- Caching semántico (embedding cache)
- Context compression (resúmenes jerárquicos)

---

## 🧪 Evaluación y Métricas

La evaluación consistente evita regresiones.

Tipos:
- Automática: BLEU, ROUGE (limitado), BERTScore, COMET.
- Basada en LLM-as-a-Judge: pares A/B, escalas Likert.
- Métricas específicas: 
  - Context hit rate (RAG)
  - Hallucination rate
  - Latencia P50/P95/P99
  - Costo por 1K tokens útiles
  - Tasa de tool success (agentes)

Herramientas:
- [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Gaia / MMLU / GSM8K] (benchmarks)
- [Promptfoo](https://github.com/promptfoo/promptfoo)
- [WeightWatcher] (calidad modelos)

---

## 🔐 Seguridad, Alineación y Riesgos

Aspectos:
- Prompt Injection
- Data Exfiltration
- Jailbreaks
- Leakage de PII
- Output Filtering / Red Teaming

Mitigaciones:
- Sanitización de entradas
- Separación de roles (system vs user)
- Clasificadores de seguridad (moderation endpoints)
- Guardrails: [NeMo Guardrails], [Guardrails-AI], [Azure Content Filters]

---

## 💰 Optimización de Costos

Estrategias:
- Seleccionar modelo por tarea (routing / cascadas)
- Cuantización local (4-bit, QLoRA)
- Prompt trimming y compresión semántica
- Reutilizar embeddings (cache)
- Streaming parcial
- Batch inference

KPIs:
- Tokens por intención
- Tokens contextuales redundantes
- Costo por sesión resuelta

---

## 🛰️ Observabilidad y Trazabilidad

Qué capturar:
- Prompt final compilado
- Versionado de plantillas
- Latencia end-to-end
- Tool calls y resultados
- Evaluaciones post-hoc

Herramientas:
- [LangSmith], [Weights & Biases], [Arize](https://arize.com), [Helicone], [E2B sandbox], [PromptLayer]

---

## 🧪 Testing de Prompts

Tipos:
- Regresión (snapshot expected outputs)
- Sensibilidad (mutaciones adversarias)
- Robustez (ruido / reorder)
- Factualidad (verificador externo)

Pipeline:
1. Dataset representativo
2. Definición de aserciones (regex, JSON schema, LLM judge)
3. Score agregado (>= umbral)
4. Gate CI/CD

---

## 🧵 Gestión de Contexto Extendido

Técnicas:
- Chunking semántico adaptativo
- Resúmenes jerárquicos (map → reduce → refine)
- Sliding window + focal retrieval
- Embedding + graph enrichment
- Long-context distillation (partial fine-tune)

Riesgos: context rot, dilución de señal, latencia.

---

## 🛣️ Roadmap de Aprendizaje Sugerido

1. Fundamentos: prompts + inferencia local (Ollama)
2. RAG básico
3. Evaluación y métricas
4. Fine-tuning PEFT
5. Tool use / agentes
6. Optimización costo-rendimiento
7. Observabilidad + seguridad
8. Orquestación avanzada (LangGraph / DSPy)

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
- [Documentación oficial de Ollama](https://github.com/ollama/ollama/tree/main/docs)
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
- [🤗 Hugging Face Hub](https://huggingface.co/models) – Repositorio central de modelos preentrenados y datasets para IA generativa, NLP, visión y más.
- [🤗 Hugging Face Trending Papers](https://huggingface.co/papers/trending) - Lista de papers recientes en IA y machine learning.
- [📝 The Gradient](https://thegradient.pub/) – Artículos y análisis sobre LLMs.
- [🧠 Awesome LLM](https://github.com/Hannibal046/Awesome-LLM) – Lista curada de modelos, datasets y herramientas.

### ⚙️ Frameworks y Librerías
- [📈 Pydantic](https://docs.pydantic.dev/latest/) - La librería de validación de datos más utilizada en Python
    - [🐙 GitHub Repository](https://github.com/pydantic/pydantic)
- [🧪 LangChain](https://www.langchain.com/) – Orquestación de agentes y flujos con LLMs, APIs y herramientas externas.
- [🔁 LangGraph](https://www.langgraph.dev/) – Framework para flujos conversacionales multiestado con LLMs.
- [📦 LlamaIndex](https://www.llamaindex.ai/) – Framework para crear aplicaciones de RAG (Retrieval-Augmented Generation).
- [🤗 Transformers (Hugging Face)](https://huggingface.co/docs/transformers/index) – Librería para el uso de modelos de lenguaje en Python.
- [LangSmith](https://www.langchain.com/langsmith) – Observabilidad
- [Helicone](https://www.helicone.ai/) – Logging de llamadas LLM
- [Promptfoo](https://github.com/promptfoo/promptfoo) – Testing de prompts
- [Guardrails AI](https://github.com/guardrails-ai/guardrails) – Validación estructural
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) – Seguridad conversacional

### Personas de Interes
- [Jeremy Howard](https://jeremy.fast.ai/)
    - [🐙 GitHub Repository](https://github.com/jph00)
    - [🐙 Fast.ai GitHub Repository](https://github.com/fastai)
    - [fast.ai—Making neural nets uncool again](https://www.fast.ai/)
- [Andrej Karpathy](https://karpathy.ai/)
    - [🐙 GitHub Repository](https://github.com/karpathy)
    - [🤗 Hugging Face Page](https://huggingface.co/karpathy)
    - X.com: [@karpathy](https://x.com/karpathy)
- [Sebastian Raschka](https://sebastianraschka.com/)
    - X.com: [@rasbt](https://x.com/rasbt)
- [Maxime Labonne](https://mlabonne.github.io/blog/)
    - [🐙 GitHub Repository](https://github.com/mlabonne)
    - [🤗 Hugging Face Page](https://huggingface.co/mlabonne)
    - [LLM Engineer's Handbook](https://github.com/PacktPublishing/LLM-Engineers-Handbook)
- [Colin Kealty (Bartowski)](https://x.com/bartowski1182)
    - [🐙 GitHub Repository](https://github.com/bartowski1182)
    - [🤗 Hugging Face Page](https://huggingface.co/bartowski)
    - [🤗 Hugging Face LM Studio Community Page](https://huggingface.co/lmstudio-community)
- [David Kim](https://x.com/interpreter_ai)
    - [🐙 GitHub Repository](https://github.com/davidkimai)
    - [🤗 Hugging Face Recursive Labs Page](https://huggingface.co/recursivelabsai)
- [Drew Breunig](https://www.dbreunig.com/)

### Enlaces Útiles
- [LLM Visualization](https://bbycroft.net/llm)
- [BertViz Interactive Tutorial](https://colab.research.google.com/drive/1hXIQ77A4TYS4y3UthWF-Ci7V7vVUoxmQ)
- nanoGPT: [🐙 GitHub Repository](https://github.com/karpathy/nanoGPT)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [The GAN Zoo](https://github.com/hindupuravinash/the-gan-zoo)