"""
app.py — Chatbot RAG sobre Gemini con Streamlit.

    streamlit run app.py

Pestaña 1: carga de documentos, estrategia de troceado y construcción del índice.
Pestaña 2: consulta con reescritura múltiple de la pregunta y fusión de resultados.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import rag_core as rc

st.set_page_config(page_title="RAG con Gemini", page_icon="🔎", layout="wide")

# --------------------------------------------------------------------------------------
# Estado
# --------------------------------------------------------------------------------------

for clave, valor in {
    "indice": None,
    "historial": [],          # [{"pregunta", "respuesta", "diag"}]
    "documentos": {},         # nombre -> texto
    "previsualizacion": None,
}.items():
    st.session_state.setdefault(clave, valor)


@st.cache_resource(show_spinner=False)
def obtener_cliente() -> rc.ClienteGemini:
    return rc.ClienteGemini()


@st.cache_data(ttl=3600, show_spinner=False)
def obtener_modelos() -> tuple[list[str], list[str]]:
    return obtener_cliente().listar_modelos()


def escoger(opciones: list[str], preferido: str) -> int:
    """Índice del modelo preferido dentro de la lista, o 0 si ya no existe."""
    if preferido in opciones:
        return opciones.index(preferido)
    for i, o in enumerate(opciones):
        if preferido.split("-")[1:2] and preferido.split("-")[1] in o:
            return i
    return 0


# --------------------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------------------

with st.sidebar:
    st.title("🔎 RAG con Gemini")

    if not rc.API_KEY:
        st.error("Falta `GEMINI_API_KEY` en el fichero `.env`.")
        st.code("GEMINI_API_KEY=tu_clave_aqui", language="bash")
        st.caption("Consíguela en aistudio.google.com/apikey")
        st.stop()

    try:
        modelos_gen, modelos_emb = obtener_modelos()
    except Exception as e:
        st.error(f"No se puede conectar con la API: {e}")
        st.stop()

    st.success(f"Conectado · {len(modelos_gen)} modelos disponibles")

    st.subheader("Modelos")
    modelo_chat = st.selectbox(
        "Generación de respuestas", modelos_gen,
        index=escoger(modelos_gen, rc.MODELO_CHAT),
        help="El que redacta la respuesta final a partir de los fragmentos.")
    modelo_aux = st.selectbox(
        "Tareas auxiliares", modelos_gen,
        index=escoger(modelos_gen, rc.MODELO_AUX),
        help="Reescritura de la pregunta y reordenado. Conviene uno rápido y barato: "
             "se le llama varias veces por consulta.")
    modelo_emb = st.selectbox(
        "Embeddings", modelos_emb, index=escoger(modelos_emb, rc.MODELO_EMBEDDING),
        help="Cambiarlo obliga a reconstruir el índice entero.")
    dim = st.select_slider(
        "Dimensión del embedding", options=[128, 256, 512, 768, 1536, 3072],
        value=rc.DIM_EMBEDDING,
        help="El modelo admite truncar el vector (Matryoshka). Menos dimensiones = "
             "menos memoria y búsqueda más rápida, con una pérdida de calidad pequeña "
             "hasta 768.")

    st.divider()
    st.subheader("Consumo de esta sesión")
    cli = obtener_cliente()
    c1, c2 = st.columns(2)
    c1.metric("Llamadas API", cli.llamadas)
    c2.metric("Tokens salida", f"{cli.tokens_salida:,}")
    c1.metric("Tokens entrada", f"{cli.tokens_entrada:,}")
    c2.metric("Tokens embed.", f"~{cli.tokens_embedding:,}")

    if st.session_state.indice:
        st.divider()
        m = st.session_state.indice.meta
        st.caption(f"**Índice activo:** `{st.session_state.indice.nombre}`  \n"
                   f"{m.get('n_trozos', 0)} trozos · {m.get('estrategia')} · "
                   f"dim {m.get('dimension')}")


# --------------------------------------------------------------------------------------
# Pestañas
# --------------------------------------------------------------------------------------

tab_index, tab_chat = st.tabs(["📥 **Indexación**", "💬 **Consulta**"])

# ======================================================================== PESTAÑA 1
with tab_index:
    st.header("Construir el índice")

    col_carga, col_guardados = st.columns([3, 2])

    with col_carga:
        st.subheader("1 · Documentos")
        subidos = st.file_uploader(
            "Arrastra tus ficheros", type=["pdf", "md", "txt", "markdown", "csv", "json"],
            accept_multiple_files=True)

        carpeta = st.text_input(
            "…o indica una carpeta local", placeholder="./corpus",
            help="Se leen de forma recursiva los ficheros con extensión soportada.")

        if st.button("Cargar documentos", type="secondary", width="stretch"):
            docs: dict[str, str] = {}
            errores: list[str] = []
            for f in subidos or []:
                try:
                    docs[f.name] = rc.leer_fichero(f.name, f.getvalue())
                except Exception as e:
                    errores.append(f"{f.name}: {e}")
            if carpeta.strip():
                base = Path(carpeta.strip()).expanduser()
                if not base.exists():
                    errores.append(f"La carpeta {base} no existe")
                else:
                    for ruta in sorted(base.rglob("*")):
                        if ruta.suffix.lower() in rc.EXTENSIONES and ruta.is_file():
                            try:
                                docs[ruta.name] = rc.leer_fichero(ruta.name, ruta.read_bytes())
                            except Exception as e:
                                errores.append(f"{ruta.name}: {e}")

            vacios = [n for n, t in docs.items() if len(t.strip()) < 50]
            st.session_state.documentos = {n: t for n, t in docs.items() if n not in vacios}
            st.session_state.previsualizacion = None
            if vacios:
                st.warning("Sin texto extraíble (¿PDF escaneado? haría falta OCR): "
                           + ", ".join(vacios))
            for e in errores:
                st.error(e)
            if st.session_state.documentos:
                st.success(f"{len(st.session_state.documentos)} documentos cargados")

        if st.session_state.documentos:
            resumen = pd.DataFrame([
                {"documento": n, "caracteres": len(t), "tokens aprox.": rc.estimar_tokens(t)}
                for n, t in st.session_state.documentos.items()])
            st.dataframe(resumen, width="stretch", hide_index=True)

    with col_guardados:
        st.subheader("Índices guardados")
        guardados = rc.Indice.listar()
        if guardados:
            elegido = st.selectbox("Disponibles", guardados, label_visibility="collapsed")
            b1, b2 = st.columns(2)
            if b1.button("Cargar", width="stretch"):
                st.session_state.indice = rc.Indice.cargar(elegido)
                st.session_state.historial = []
                st.success(f"Índice «{elegido}» cargado")
                st.rerun()
            if b2.button("Borrar", width="stretch"):
                import shutil
                shutil.rmtree(rc.DIR_INDICES / elegido, ignore_errors=True)
                st.rerun()
        else:
            st.caption("Todavía no hay ninguno. Construye uno y guárdalo para no "
                       "volver a pagar los embeddings.")

    st.divider()
    st.subheader("2 · Estrategia de troceado")

    col_est, col_par = st.columns([2, 3])
    with col_est:
        estrategia = st.radio(
            "Cómo se corta el documento", list(rc.ESTRATEGIAS_CHUNKING),
            format_func=lambda k: k.capitalize(),
            captions=list(rc.ESTRATEGIAS_CHUNKING.values()))

    with col_par:
        c1, c2 = st.columns(2)
        tam = c1.slider("Tamaño del trozo (caracteres)", 200, 3000, 900, 50)
        solape = c2.slider("Solape", 0, 600, 150, 10,
                           help="Para que una idea partida por el corte siga siendo "
                                "recuperable completa desde alguno de los dos trozos.")
        contextual = st.toggle(
            "Prefijo contextual en cada trozo", value=True,
            help="Antepone «[documento · apartado]» al texto antes de calcular el "
                 "embedding. Dos líneas de código y de las mejoras más rentables que existen.")
        if estrategia == "semantico":
            st.info("El troceado semántico embebe todas las frases del corpus para "
                    "decidir dónde cortar: cuesta bastante más al indexar.", icon="💸")
        if solape >= tam:
            st.error("El solape debe ser menor que el tamaño del trozo.")

    if st.session_state.documentos and estrategia != "semantico":
        if st.button("👁️ Previsualizar troceado (no consume API)"):
            vista = []
            for nombre, texto in st.session_state.documentos.items():
                vista.extend(rc.trocear_documento(nombre, texto, estrategia, tam, solape))
            st.session_state.previsualizacion = vista

    if st.session_state.previsualizacion:
        trozos = st.session_state.previsualizacion
        longitudes = [len(t.texto) for t in trozos]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trozos", len(trozos))
        m2.metric("Longitud media", f"{int(np.mean(longitudes))}")
        m3.metric("Longitud máxima", f"{max(longitudes)}")
        m4.metric("Tokens a embeber", f"~{sum(rc.estimar_tokens(t.texto) for t in trozos):,}")

        with st.expander("Ver los trozos generados"):
            st.dataframe(pd.DataFrame([
                {"id": t.id, "apartado": t.seccion, "chars": len(t.texto),
                 "texto": t.texto[:300].replace("\n", " ")} for t in trozos]),
                width="stretch", hide_index=True, height=300)
            st.caption("Mira si alguna tabla ha quedado partida por la mitad: es el "
                       "fallo que después se manifiesta como un cálculo inventado.")

    st.divider()
    st.subheader("3 · Construir")

    col_nom, col_btn = st.columns([3, 1])
    nombre_idx = col_nom.text_input("Nombre del índice",
                                    value=f"indice_{time.strftime('%Y%m%d_%H%M')}")
    construir = col_btn.button("⚙️ Construir índice", type="primary",
                               width="stretch",
                               disabled=not st.session_state.documentos or solape >= tam)

    if construir:
        barra = st.progress(0.0, text="Preparando…")
        try:
            indice = rc.construir_indice(
                nombre_idx, st.session_state.documentos, obtener_cliente(),
                estrategia=estrategia, tam=tam, solape=solape, contextual=contextual,
                modelo_emb=modelo_emb, dim=dim,
                progreso=lambda msg, frac: barra.progress(min(frac, 1.0), text=msg))
            indice.guardar()
            st.session_state.indice = indice
            st.session_state.historial = []
            barra.empty()
            st.success(f"Índice construido y guardado en `{rc.DIR_INDICES / nombre_idx}`")
            st.json(indice.meta, expanded=False)
        except Exception as e:
            barra.empty()
            st.exception(e)

# ======================================================================== PESTAÑA 2
with tab_chat:
    if st.session_state.indice is None:
        st.info("Carga o construye un índice en la pestaña de **Indexación** para empezar.",
                icon="👈")
        st.stop()

    with st.expander("⚙️ Estrategia de recuperación", expanded=False):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Reescritura de la pregunta**")
            n_variantes = st.slider(
                "Variantes generadas", 0, 6, 3,
                help="La búsqueda no depende de una sola formulación: se generan N "
                     "reescrituras, se busca con todas y se fusionan los rankings.")
            hyde = st.toggle("Añadir HyDE", value=False,
                             help="El modelo redacta una respuesta hipotética y se busca "
                                  "también con SU embedding. Funciona bien en dominios técnicos.")
            condensar = st.toggle("Condensar con el historial", value=True,
                                  help="Resuelve referencias como «¿y en el otro caso?» "
                                       "antes de buscar.")

        with c2:
            st.markdown("**Búsqueda**")
            hibrido = st.toggle("Híbrida (densa + BM25)", value=True,
                                help="La densa entiende sinónimos; BM25 encuentra códigos "
                                     "e identificadores exactos. Se fusionan con RRF.")
            k_candidatos = st.slider("Candidatos por ranking", 5, 50, 20)
            k_final = st.slider("Fragmentos al modelo", 1, 15, 5,
                                help="Más allá de 10 suele empeorar: el modelo atiende "
                                     "peor a lo que queda en el centro del contexto.")

        with c3:
            st.markdown("**Filtrado final**")
            modo_final = st.radio(
                "Cómo se eligen los definitivos",
                ["Reordenado con LLM", "MMR (diversidad)", "Solo RRF"],
                help="El reordenado es lo más preciso y añade una llamada. "
                     "MMR penaliza fragmentos redundantes entre sí.")
            temperatura = st.slider("Temperatura de la respuesta", 0.0, 1.0, 0.0, 0.1)
            mostrar_fuentes = st.toggle("Mostrar fragmentos recuperados", value=True)

    st.caption(f"Índice **{st.session_state.indice.nombre}** · "
               f"{len(st.session_state.indice.trozos)} trozos · "
               f"{st.session_state.indice.meta.get('n_documentos', '?')} documentos")

    # ---------------------------------------------------------------- historial
    for turno in st.session_state.historial:
        with st.chat_message("user"):
            st.markdown(turno["pregunta"])
        with st.chat_message("assistant"):
            st.markdown(turno["respuesta"])
            if turno.get("diag"):
                _mostrar = turno["diag"]
                with st.expander("🔍 Cómo se ha construido esta respuesta"):
                    st.markdown("**Consultas lanzadas contra el índice**")
                    for v in _mostrar["variantes"]:
                        st.markdown(f"- {v}")
                    st.markdown("**Fragmentos usados**")
                    st.dataframe(pd.DataFrame(_mostrar["fragmentos"]),
                                 width="stretch", hide_index=True)
                    st.caption(_mostrar["tiempos"])

    # ---------------------------------------------------------------- nueva pregunta
    pregunta = st.chat_input("Pregunta lo que quieras sobre tus documentos")

    if pregunta:
        with st.chat_message("user"):
            st.markdown(pregunta)

        historial_tuplas = [(t["pregunta"], t["respuesta"]) for t in st.session_state.historial]

        with st.chat_message("assistant"):
            try:
                with st.status("Recuperando…", expanded=False) as estado:
                    t0 = time.time()
                    estado.update(label="Reescribiendo la pregunta…")
                    rec = rc.recuperar(
                        obtener_cliente(), st.session_state.indice, pregunta,
                        n_variantes=n_variantes, hyde=hyde, hibrido=hibrido,
                        k_final=k_final, k_candidatos=k_candidatos,
                        usar_rerank=(modo_final == "Reordenado con LLM"),
                        usar_mmr=(modo_final == "MMR (diversidad)"),
                        modelo_aux=modelo_aux, historial=historial_tuplas,
                        condensar=condensar)
                    estado.update(
                        label=f"{len(rec.variantes)} consultas → {len(rec.trozos)} fragmentos "
                              f"({time.time() - t0:.1f} s)", state="complete")

                prompt = rc.construir_prompt(rec.pregunta_usada, rec.trozos, historial_tuplas)
                respuesta = st.write_stream(
                    obtener_cliente().generar_stream(
                        prompt, sistema=rc.PROMPT_SISTEMA, modelo=modelo_chat,
                        temperatura=temperatura))

                if rc.CENTINELA in respuesta:
                    st.warning("El sistema no ha encontrado la respuesta en los documentos. "
                               "Prueba a subir el número de fragmentos o de variantes.",
                               icon="🤷")

                fragmentos = [
                    {"id": t.id, "documento": t.doc, "apartado": t.seccion,
                     "RRF": p, "encontrado por": " | ".join(rec.procedencia.get(t.id, [])),
                     "texto": t.texto[:400].replace("\n", " ")}
                    for t, p in zip(rec.trozos, rec.puntuaciones)]
                diag = {
                    "variantes": rec.variantes,
                    "fragmentos": fragmentos,
                    "tiempos": " · ".join(f"{k}: {v:.2f}s" for k, v in rec.tiempos.items())
                                + f" · total: {time.time() - t0:.1f}s",
                }

                if mostrar_fuentes:
                    with st.expander("🔍 Cómo se ha construido esta respuesta", expanded=False):
                        st.markdown("**Consultas lanzadas contra el índice**")
                        if rec.pregunta_usada != pregunta:
                            st.caption(f"Pregunta condensada con el historial: "
                                       f"*{rec.pregunta_usada}*")
                        for v in rec.variantes:
                            st.markdown(f"- {v}")
                        st.markdown("**Fragmentos usados**")
                        st.dataframe(pd.DataFrame(fragmentos), width="stretch",
                                     hide_index=True)
                        st.caption(diag["tiempos"])

                st.session_state.historial.append(
                    {"pregunta": pregunta, "respuesta": respuesta, "diag": diag})

            except Exception as e:
                st.error(f"Error durante la consulta: {e}")

    if st.session_state.historial:
        if st.button("🗑️ Vaciar conversación"):
            st.session_state.historial = []
            st.rerun()
