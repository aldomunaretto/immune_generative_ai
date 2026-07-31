"""
rag_core.py — Motor RAG sobre la API de Gemini.

Contiene toda la lógica (sin Streamlit) para que se pueda probar desde consola:

    python rag_core.py --selftest      # sin llamar a la API: valida la lógica
    python rag_core.py --smoke         # con GEMINI_API_KEY: valida la conexión

Diseño: sin dependencias de frameworks de RAG. Todo lo que hace el sistema se ve
aquí, que es justo lo que interesa cuando esto se usa para enseñar.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------------------

API_KEY = os.getenv("GEMINI_API_KEY", "")
MODELO_CHAT = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash")
MODELO_AUX = os.getenv("GEMINI_AUX_MODEL", "gemini-3.5-flash-lite")
MODELO_EMBEDDING = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
DIM_EMBEDDING = int(os.getenv("GEMINI_EMBEDDING_DIM", "768"))
DIR_INDICES = Path(os.getenv("RAG_INDEX_DIR", "./indices"))

# Límites de la API de embeddings: 250 textos y 20.000 tokens por petición,
# y solo se usan los primeros 2.048 tokens de cada texto.
LOTE_EMBEDDING = int(os.getenv("RAG_EMBED_BATCH", "50"))

CENTINELA = "NO CONSTA EN LOS DOCUMENTOS"

PROMPT_SISTEMA = f"""Eres un asistente documental que responde EXCLUSIVAMENTE con la información de los fragmentos proporcionados.

Reglas de obligado cumplimiento:
1. Si la respuesta no está en los fragmentos, responde exactamente: {CENTINELA}
2. Cita entre corchetes la fuente de cada dato, con el nombre de fichero: [informe_2025.pdf]
3. Si los fragmentos se contradicen entre sí, dilo explícitamente en lugar de elegir uno.
4. No uses conocimiento propio ni rellenes huecos con suposiciones plausibles.
5. Si la pregunta da por supuesto algo que los fragmentos no confirman, corrige la premisa.
6. Responde en el idioma de la pregunta, de forma directa y sin preámbulos."""


# --------------------------------------------------------------------------------------
# Utilidades de texto
# --------------------------------------------------------------------------------------

_STOP_ES = set("""a al algo algunas algunos ante antes como con contra cual cuando de del desde donde dos
el ella ellas ellos en entre era erais eran eres es esa esas ese eso esos esta estas este esto estos ha
hasta hay la las le les lo los mas más me mi mis mucho muy nada ni no nos nuestra nuestro o os otra otro
para pero poco por porque que quien se sea si sin sobre son su sus también tanto te tiene todo todos tu
tus un una uno unos y ya""".split())


def normalizar(texto: str) -> str:
    """Minúsculas sin acentos, para comparaciones robustas."""
    t = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def tokenizar(texto: str) -> list[str]:
    """Tokenizador para BM25: conserva códigos tipo HLS-4093 y descarta vacíos."""
    tokens = re.findall(r"[a-z0-9][a-z0-9\-_.]*", normalizar(texto))
    return [t for t in tokens if t not in _STOP_ES and len(t) > 1]


def estimar_tokens(texto: str) -> int:
    """Aproximación barata: ~4 caracteres por token. Suficiente para presupuestar."""
    return max(1, len(texto) // 4)


# --------------------------------------------------------------------------------------
# Cliente Gemini
# --------------------------------------------------------------------------------------

class ClienteGemini:
    """Envoltorio fino sobre google-genai con reintentos y contabilidad de tokens."""

    def __init__(self, api_key: str | None = None):
        from google import genai  # import perezoso: el selftest no lo necesita

        clave = api_key or API_KEY
        if not clave:
            raise RuntimeError(
                "Falta GEMINI_API_KEY. Créala en https://aistudio.google.com/apikey "
                "y ponla en el fichero .env"
            )
        self._genai = genai
        self.client = genai.Client(api_key=clave)
        self.tokens_entrada = 0
        self.tokens_salida = 0
        self.tokens_embedding = 0
        self.llamadas = 0

    # ---------------------------------------------------------------- infraestructura
    @staticmethod
    def _reintentar(fn: Callable, intentos: int = 5, espera: float = 2.0):
        """Retroceso exponencial ante 429 (cuota) y 5xx (transitorios)."""
        ultimo = None
        for i in range(intentos):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001 — el SDK envuelve varios tipos
                ultimo = e
                msg = str(e)
                recuperable = any(s in msg for s in
                                  ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE",
                                   "500", "INTERNAL", "504", "DEADLINE"))
                if not recuperable or i == intentos - 1:
                    raise
                time.sleep(espera * (2 ** i))
        raise ultimo  # pragma: no cover

    def listar_modelos(self) -> tuple[list[str], list[str]]:
        """Devuelve (modelos de generación, modelos de embedding) disponibles."""
        try:
            modelos = list(self.client.models.list())
        except Exception:
            return ([MODELO_CHAT, MODELO_AUX], [MODELO_EMBEDDING])
        gen, emb = [], []
        for m in modelos:
            nombre = (getattr(m, "name", "") or "").replace("models/", "")
            if not nombre:
                continue
            acciones = [a.lower() for a in (getattr(m, "supported_actions", None) or [])]
            if "embed" in nombre or "embedcontent" in acciones:
                emb.append(nombre)
            elif not acciones or "generatecontent" in acciones:
                if any(p in nombre for p in ("gemini", "gemma")) and "image" not in nombre \
                        and "tts" not in nombre and "embedding" not in nombre:
                    gen.append(nombre)
        return (sorted(set(gen)) or [MODELO_CHAT],
                sorted(set(emb)) or [MODELO_EMBEDDING])

    # ---------------------------------------------------------------------- embeddings
    def embeber(self, textos: Sequence[str], *, consulta: bool = False,
                modelo: str | None = None, dim: int | None = None,
                progreso: Callable[[int, int], None] | None = None) -> np.ndarray:
        """
        Devuelve una matriz (n, dim) de vectores NORMALIZADOS.

        `task_type` distinto para documentos y consultas: el modelo proyecta cada uno
        en el espacio del otro. Omitirlo degrada la calidad de forma silenciosa.
        """
        from google.genai import types

        modelo = modelo or MODELO_EMBEDDING
        dim = dim or DIM_EMBEDDING
        tarea = "RETRIEVAL_QUERY" if consulta else "RETRIEVAL_DOCUMENT"
        salida: list[list[float]] = []

        for i in range(0, len(textos), LOTE_EMBEDDING):
            lote = [t[:8000] for t in textos[i:i + LOTE_EMBEDDING]]  # ~2.048 tokens
            resp = self._reintentar(lambda: self.client.models.embed_content(
                model=modelo,
                contents=lote,
                config=types.EmbedContentConfig(task_type=tarea, output_dimensionality=dim),
            ))
            self.llamadas += 1
            self.tokens_embedding += sum(estimar_tokens(t) for t in lote)
            salida.extend(e.values for e in resp.embeddings)
            if progreso:
                progreso(min(i + LOTE_EMBEDDING, len(textos)), len(textos))

        m = np.asarray(salida, dtype="float32")
        # Con output_dimensionality < 3072 el vector llega truncado (Matryoshka) y deja
        # de tener norma 1: hay que renormalizar o el coseno queda sesgado.
        normas = np.linalg.norm(m, axis=1, keepdims=True)
        normas[normas == 0] = 1.0
        return m / normas

    # ---------------------------------------------------------------------- generación
    def generar(self, prompt: str, *, sistema: str | None = None, modelo: str | None = None,
                temperatura: float = 0.0, max_tokens: int = 1500,
                json_schema=None, sin_razonamiento: bool = False) -> str:
        from google.genai import types

        cfg: dict = {"temperature": temperatura, "max_output_tokens": max_tokens}
        if sistema:
            cfg["system_instruction"] = sistema
        if json_schema is not None:
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = json_schema
        if sin_razonamiento:
            # Ahorra latencia y tokens en las llamadas auxiliares (reescritura, rerank).
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

        def _llamar(config: dict):
            return self.client.models.generate_content(
                model=modelo or MODELO_CHAT,
                contents=prompt,
                config=types.GenerateContentConfig(**config),
            )

        try:
            resp = self._reintentar(lambda: _llamar(cfg))
        except Exception as e:
            # Algunos modelos no aceptan thinking_config o response_schema: reintenta limpio.
            if "thinking" in str(e).lower() or "schema" in str(e).lower():
                cfg.pop("thinking_config", None)
                resp = self._reintentar(lambda: _llamar(cfg))
            else:
                raise

        self.llamadas += 1
        uso = getattr(resp, "usage_metadata", None)
        if uso:
            self.tokens_entrada += getattr(uso, "prompt_token_count", 0) or 0
            self.tokens_salida += getattr(uso, "candidates_token_count", 0) or 0
        return (resp.text or "").strip()

    def generar_stream(self, prompt: str, *, sistema: str | None = None,
                       modelo: str | None = None, temperatura: float = 0.0,
                       max_tokens: int = 1500) -> Iterable[str]:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            temperature=temperatura, max_output_tokens=max_tokens,
            system_instruction=sistema)
        flujo = self.client.models.generate_content_stream(
            model=modelo or MODELO_CHAT, contents=prompt, config=cfg)
        self.llamadas += 1
        for parte in flujo:
            uso = getattr(parte, "usage_metadata", None)
            if uso:
                self.tokens_entrada = max(self.tokens_entrada,
                                          getattr(uso, "prompt_token_count", 0) or 0)
                self.tokens_salida += getattr(uso, "candidates_token_count", 0) or 0
            if parte.text:
                yield parte.text


# --------------------------------------------------------------------------------------
# Troceado
# --------------------------------------------------------------------------------------

@dataclass
class Trozo:
    id: str
    doc: str
    texto: str
    seccion: str = ""
    orden: int = 0

    def con_contexto(self) -> str:
        """Texto tal y como se indexa cuando el prefijo contextual está activo."""
        cabecera = f"[{self.doc}" + (f" · {self.seccion}" if self.seccion else "") + "]"
        return f"{cabecera}\n{self.texto}"


ESTRATEGIAS_CHUNKING = {
    "ventana": "Ventana fija — N caracteres con solape. Simple y universal.",
    "recursivo": "Recursivo — corta por párrafo, luego frase, luego carácter. El estándar razonable.",
    "estructural": "Estructural — respeta apartados y encabezados. El mejor si hay tablas.",
    "semantico": "Semántico — corta donde cambia el tema. Consume embeddings al indexar.",
}


def _ventana(texto: str, tam: int, solape: int) -> list[str]:
    paso = max(1, tam - solape)
    trozos, i = [], 0
    while i < len(texto):
        fin = min(i + tam, len(texto))
        trozos.append(texto[i:fin])
        if fin == len(texto):
            break
        i += paso
    return [t.strip() for t in trozos if t.strip()]


def _recursivo(texto: str, tam: int, solape: int,
               separadores: tuple[str, ...] = ("\n\n", "\n", ". ", " ")) -> list[str]:
    """Divide por el separador de mayor nivel que consiga trozos por debajo de `tam`."""
    if len(texto) <= tam:
        return [texto.strip()] if texto.strip() else []
    if not separadores:
        return _ventana(texto, tam, solape)

    sep, resto = separadores[0], separadores[1:]
    piezas = texto.split(sep)
    # El solape se añade después, así que se acumula hasta `tam - solape`:
    # de este modo el trozo final nunca supera `tam`.
    objetivo = max(1, tam - solape)
    salida, actual = [], ""
    for pieza in piezas:
        candidato = (actual + sep + pieza) if actual else pieza
        if len(candidato) <= objetivo:
            actual = candidato
        else:
            if actual:
                salida.append(actual)
            actual = pieza if len(pieza) <= tam else ""
            if not actual:
                salida.extend(_recursivo(pieza, tam, solape, resto))
    if actual:
        salida.append(actual)

    # Solape entre trozos consecutivos, tomando la cola del anterior.
    if solape > 0 and len(salida) > 1:
        con_solape = [salida[0]]
        for prev, act in zip(salida, salida[1:]):
            con_solape.append((prev[-solape:] + " " + act).strip())
        salida = con_solape
    return [t.strip() for t in salida if t.strip()]


_RE_SECCION = re.compile(r"^(#{1,4}\s+.+|\d{1,2}(\.\d{1,2})*\.?\s+[^\n]{3,90}|[A-ZÁÉÍÓÚÑ][^\n]{3,70}:)\s*$",
                         re.MULTILINE)


def _estructural(texto: str, tam: int, solape: int) -> list[tuple[str, str]]:
    """Devuelve (seccion, texto). Corta por encabezados y respeta el tamaño máximo."""
    marcas = [(m.start(), m.group().strip()) for m in _RE_SECCION.finditer(texto)]
    if not marcas:
        return [("", t) for t in _recursivo(texto, tam, solape)]

    if marcas[0][0] > 0:
        marcas.insert(0, (0, ""))
    salida: list[tuple[str, str]] = []
    for idx, (ini, titulo) in enumerate(marcas):
        fin = marcas[idx + 1][0] if idx + 1 < len(marcas) else len(texto)
        cuerpo = texto[ini:fin].strip()
        if not cuerpo:
            continue
        limpio = re.sub(r"^#{1,4}\s+", "", titulo).strip(": ").strip()
        partes = [cuerpo] if len(cuerpo) <= tam else _recursivo(cuerpo, tam, solape)
        salida.extend((limpio, p) for p in partes)

    # Un apartado muy corto no da contexto suficiente para responder: se funden los
    # consecutivos hasta acercarse a `tam`, conservando el título del primero.
    minimo = max(120, tam // 4)
    fundidos: list[tuple[str, str]] = []
    for seccion, cuerpo in salida:
        if fundidos and len(fundidos[-1][1]) < minimo and \
                len(fundidos[-1][1]) + len(cuerpo) + 1 <= tam:
            prev_sec, prev_txt = fundidos[-1]
            fundidos[-1] = (prev_sec or seccion, f"{prev_txt}\n{cuerpo}")
        else:
            fundidos.append((seccion, cuerpo))
    return fundidos


def _semantico(texto: str, tam: int, embeber: Callable[[Sequence[str]], np.ndarray],
               percentil: int = 88) -> list[str]:
    """
    Corta donde la similitud entre frases consecutivas cae por debajo del percentil.
    Cuesta una pasada de embeddings sobre todas las frases: úsalo con criterio.
    """
    frases = [f.strip() for f in re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡])", texto) if f.strip()]
    if len(frases) < 4:
        return [texto.strip()] if texto.strip() else []

    vec = embeber(frases)
    dist = 1.0 - np.sum(vec[:-1] * vec[1:], axis=1)
    if len(dist) == 0:
        return [texto]
    umbral = float(np.percentile(dist, percentil))

    trozos, actual = [], [frases[0]]
    for i, frase in enumerate(frases[1:]):
        largo = sum(len(f) for f in actual)
        if dist[i] > umbral or largo + len(frase) > tam:
            trozos.append(" ".join(actual))
            actual = [frase]
        else:
            actual.append(frase)
    if actual:
        trozos.append(" ".join(actual))
    return [t for t in trozos if t.strip()]


def trocear_documento(nombre: str, texto: str, estrategia: str, tam: int, solape: int,
                      embeber: Callable[[Sequence[str]], np.ndarray] | None = None) -> list[Trozo]:
    """Aplica la estrategia elegida y devuelve objetos Trozo listos para indexar."""
    if estrategia == "estructural":
        pares = _estructural(texto, tam, solape)
    elif estrategia == "semantico":
        if embeber is None:
            raise ValueError("El troceado semántico necesita una función de embeddings")
        pares = [("", t) for t in _semantico(texto, tam, embeber)]
    elif estrategia == "recursivo":
        pares = [("", t) for t in _recursivo(texto, tam, solape)]
    elif estrategia == "ventana":
        pares = [("", t) for t in _ventana(texto, tam, solape)]
    else:
        raise ValueError(f"Estrategia desconocida: {estrategia}")

    return [Trozo(id=f"{nombre}#{i}", doc=nombre, texto=t, seccion=s, orden=i)
            for i, (s, t) in enumerate(pares)]


# --------------------------------------------------------------------------------------
# Índice
# --------------------------------------------------------------------------------------

@dataclass
class Indice:
    nombre: str
    trozos: list[Trozo] = field(default_factory=list)
    matriz: np.ndarray | None = None
    meta: dict = field(default_factory=dict)
    _bm25 = None

    # ------------------------------------------------------------------ búsqueda densa
    def buscar_denso(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self.matriz is None or len(self.trozos) == 0:
            return []
        sims = self.matriz @ vector          # coseno: todo está normalizado
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i])) for i in idx]

    # ------------------------------------------------------------------ búsqueda léxica
    @property
    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([tokenizar(t.texto) for t in self.trozos])
        return self._bm25

    def buscar_lexico(self, consulta: str, k: int) -> list[tuple[int, float]]:
        if not self.trozos:
            return []
        notas = self.bm25.get_scores(tokenizar(consulta))
        k = min(k, len(notas))
        idx = np.argpartition(-notas, k - 1)[:k]
        idx = idx[np.argsort(-notas[idx])]
        return [(int(i), float(notas[i])) for i in idx]

    # ----------------------------------------------------------------- persistencia
    def guardar(self, directorio: Path | None = None) -> Path:
        destino = (directorio or DIR_INDICES) / self.nombre
        destino.mkdir(parents=True, exist_ok=True)
        np.save(destino / "matriz.npy", self.matriz)
        with open(destino / "trozos.pkl", "wb") as f:
            pickle.dump([asdict(t) for t in self.trozos], f)
        (destino / "meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return destino

    @classmethod
    def cargar(cls, nombre: str, directorio: Path | None = None) -> "Indice":
        origen = (directorio or DIR_INDICES) / nombre
        with open(origen / "trozos.pkl", "rb") as f:
            trozos = [Trozo(**d) for d in pickle.load(f)]
        meta = json.loads((origen / "meta.json").read_text(encoding="utf-8"))
        return cls(nombre=nombre, trozos=trozos,
                   matriz=np.load(origen / "matriz.npy"), meta=meta)

    @staticmethod
    def listar(directorio: Path | None = None) -> list[str]:
        base = directorio or DIR_INDICES
        if not base.exists():
            return []
        return sorted(d.name for d in base.iterdir()
                      if d.is_dir() and (d / "matriz.npy").exists())


def construir_indice(nombre: str, documentos: dict[str, str], cliente: ClienteGemini, *,
                     estrategia: str = "recursivo", tam: int = 900, solape: int = 150,
                     contextual: bool = True, modelo_emb: str | None = None,
                     dim: int | None = None,
                     progreso: Callable[[str, float], None] | None = None) -> Indice:
    """Trocea, embebe e indexa. `progreso(mensaje, fraccion)` alimenta la barra de la UI."""
    t0 = time.time()
    embeber_frases = (lambda fs: cliente.embeber(fs, consulta=False,
                                                 modelo=modelo_emb, dim=dim))

    trozos: list[Trozo] = []
    for i, (nombre_doc, texto) in enumerate(documentos.items()):
        if progreso:
            progreso(f"Troceando {nombre_doc}", 0.15 * (i + 1) / max(1, len(documentos)))
        trozos.extend(trocear_documento(nombre_doc, texto, estrategia, tam, solape,
                                        embeber_frases if estrategia == "semantico" else None))

    if not trozos:
        raise ValueError("No se ha generado ningún trozo: ¿los documentos tienen texto extraíble?")

    textos = [t.con_contexto() if contextual else t.texto for t in trozos]

    def _avance(hecho: int, total: int):
        if progreso:
            progreso(f"Calculando embeddings ({hecho}/{total} trozos)",
                     0.15 + 0.85 * hecho / total)

    matriz = cliente.embeber(textos, consulta=False, modelo=modelo_emb, dim=dim,
                             progreso=_avance)

    longitudes = [len(t.texto) for t in trozos]
    meta = {
        "creado": time.strftime("%Y-%m-%d %H:%M"),
        "documentos": list(documentos),
        "n_documentos": len(documentos),
        "n_trozos": len(trozos),
        "estrategia": estrategia,
        "tam": tam,
        "solape": solape,
        "contextual": contextual,
        "modelo_embedding": modelo_emb or MODELO_EMBEDDING,
        "dimension": int(matriz.shape[1]),
        "long_media": int(np.mean(longitudes)),
        "long_max": int(np.max(longitudes)),
        "tokens_embedding_aprox": cliente.tokens_embedding,
        "segundos": round(time.time() - t0, 1),
    }
    return Indice(nombre=nombre, trozos=trozos, matriz=matriz, meta=meta)


# --------------------------------------------------------------------------------------
# Recuperación
# --------------------------------------------------------------------------------------

ESQUEMA_LISTA = {"type": "ARRAY", "items": {"type": "STRING"}}


def condensar_pregunta(cliente: ClienteGemini, pregunta: str,
                       historial: list[tuple[str, str]], modelo: str) -> str:
    """Resuelve referencias del tipo «¿y para el otro caso?» usando el historial."""
    if not historial:
        return pregunta
    contexto = "\n".join(f"Usuario: {u}\nAsistente: {a[:300]}" for u, a in historial[-3:])
    prompt = (f"Historial de la conversación:\n{contexto}\n\n"
              f"Última pregunta del usuario: {pregunta}\n\n"
              "Reescríbela como una pregunta autónoma que se entienda sin el historial. "
              "Si ya es autónoma, devuélvela tal cual. Responde SOLO con la pregunta.")
    try:
        salida = cliente.generar(prompt, modelo=modelo, temperatura=0.0,
                                 max_tokens=200, sin_razonamiento=True)
        return salida.strip().strip('"') or pregunta
    except Exception:
        return pregunta


def generar_variantes(cliente: ClienteGemini, pregunta: str, n: int, modelo: str,
                      hyde: bool = False) -> list[str]:
    """
    Multi-consulta: n reformulaciones + (opcional) una respuesta hipotética (HyDE).

    Por qué funciona: la pregunta del usuario y el documento rara vez comparten
    vocabulario. Buscar con varias formulaciones cubre más superficie del espacio
    vectorial y la fusión posterior premia lo que aparece en varias de ellas.
    """
    variantes = [pregunta]
    if n <= 0:
        return variantes

    prompt = (
        f"Genera {n} reformulaciones distintas de esta pregunta para buscar en una base "
        f"documental.\n\nPregunta: {pregunta}\n\n"
        "Criterios:\n"
        "- Cada variante debe atacar la pregunta desde un ángulo distinto: sinónimos, "
        "terminología técnica del dominio, formulación como afirmación, versión más "
        "específica y versión más general.\n"
        "- Conserva SIEMPRE los identificadores, códigos, cifras y nombres propios literales.\n"
        "- No respondas a la pregunta, solo reformúlala.\n"
        f"Devuelve un array JSON con exactamente {n} cadenas."
    )
    try:
        crudo = cliente.generar(prompt, modelo=modelo, temperatura=0.7, max_tokens=600,
                                json_schema=ESQUEMA_LISTA, sin_razonamiento=True)
        nuevas = json.loads(crudo)
        if isinstance(nuevas, list):
            variantes += [str(v).strip() for v in nuevas if str(v).strip()][:n]
    except Exception:
        # Reserva: pedir texto plano y partir por líneas.
        try:
            crudo = cliente.generar(prompt.replace("Devuelve un array JSON", "Devuelve una por línea"),
                                    modelo=modelo, temperatura=0.7, max_tokens=400,
                                    sin_razonamiento=True)
            nuevas = [re.sub(r"^[\-\d.\)\s]+", "", l).strip()
                      for l in crudo.splitlines() if l.strip()]
            variantes += nuevas[:n]
        except Exception:
            pass

    if hyde:
        try:
            hipotetica = cliente.generar(
                f"Redacta en 3 o 4 frases el párrafo de un documento interno que "
                f"respondería a esta pregunta. Inventa el contenido con el estilo y "
                f"vocabulario de una documentación corporativa; no digas que es hipotético.\n\n"
                f"Pregunta: {pregunta}",
                modelo=modelo, temperatura=0.5, max_tokens=300, sin_razonamiento=True)
            if hipotetica:
                variantes.append(hipotetica)
        except Exception:
            pass

    # Deduplica conservando el orden (la original siempre primera).
    vistas, unicas = set(), []
    for v in variantes:
        clave = normalizar(v)[:120]
        if clave not in vistas:
            vistas.add(clave)
            unicas.append(v)
    return unicas


def fusionar_rrf(rankings: list[list[tuple[int, float]]], k_rrf: int = 60) -> dict[int, float]:
    """
    Reciprocal Rank Fusion: suma 1/(k + posición) de cada ranking.

    Fusiona por POSICIÓN, no por puntuación, así que no hay que normalizar escalas
    incompatibles (el coseno vive en [-1,1] y BM25 no tiene techo).
    """
    puntos: dict[int, float] = {}
    for ranking in rankings:
        for pos, (idx, _) in enumerate(ranking):
            puntos[idx] = puntos.get(idx, 0.0) + 1.0 / (k_rrf + pos + 1)
    return puntos


def mmr(indice: Indice, candidatos: list[int], vector_consulta: np.ndarray,
        k: int, lambda_: float = 0.7) -> list[int]:
    """Maximal Marginal Relevance: penaliza candidatos redundantes entre sí."""
    if indice.matriz is None or not candidatos:
        return candidatos[:k]
    seleccionados: list[int] = []
    restantes = list(candidatos)
    rel = {i: float(indice.matriz[i] @ vector_consulta) for i in restantes}
    while restantes and len(seleccionados) < k:
        if not seleccionados:
            mejor = max(restantes, key=lambda i: rel[i])
        else:
            def puntuar(i: int) -> float:
                redundancia = max(float(indice.matriz[i] @ indice.matriz[j])
                                  for j in seleccionados)
                return lambda_ * rel[i] - (1 - lambda_) * redundancia
            mejor = max(restantes, key=puntuar)
        seleccionados.append(mejor)
        restantes.remove(mejor)
    return seleccionados


def reordenar_llm(cliente: ClienteGemini, pregunta: str, indice: Indice,
                  candidatos: list[int], k: int, modelo: str) -> list[int]:
    """
    Reordenado con el LLM como juez de relevancia (sustituto del cross-encoder).

    Una sola llamada para todos los candidatos: es más barato y más consistente que
    puntuarlos de uno en uno.
    """
    if len(candidatos) <= k:
        return candidatos
    listado = "\n\n".join(
        f"[{n}] {indice.trozos[i].con_contexto()[:700]}"
        for n, i in enumerate(candidatos))
    prompt = (f"Pregunta: {pregunta}\n\nFragmentos candidatos:\n\n{listado}\n\n"
              f"Devuelve un array JSON con los números de los {k} fragmentos MÁS útiles "
              f"para responder la pregunta, del más útil al menos útil. Solo números, "
              f"como cadenas de texto. Si un fragmento no aporta nada, no lo incluyas.")
    try:
        crudo = cliente.generar(prompt, modelo=modelo, temperatura=0.0, max_tokens=300,
                                json_schema=ESQUEMA_LISTA, sin_razonamiento=True)
        orden = [int(re.sub(r"\D", "", str(x))) for x in json.loads(crudo)
                 if re.sub(r"\D", "", str(x)) != ""]
        elegidos = [candidatos[n] for n in orden if 0 <= n < len(candidatos)]
        vistos, limpio = set(), []
        for i in elegidos:
            if i not in vistos:
                vistos.add(i)
                limpio.append(i)
        # Rellena por si el modelo devolvió menos de k.
        for i in candidatos:
            if len(limpio) >= k:
                break
            if i not in vistos:
                limpio.append(i)
                vistos.add(i)
        return limpio[:k]
    except Exception:
        return candidatos[:k]


@dataclass
class Recuperacion:
    trozos: list[Trozo]
    puntuaciones: list[float]
    variantes: list[str]
    procedencia: dict[str, list[str]]   # id de trozo -> variantes que lo encontraron
    tiempos: dict[str, float]
    pregunta_usada: str


def recuperar(cliente: ClienteGemini, indice: Indice, pregunta: str, *,
              n_variantes: int = 3, hyde: bool = False, hibrido: bool = True,
              k_final: int = 5, k_candidatos: int = 20, usar_rerank: bool = True,
              usar_mmr: bool = False, modelo_aux: str = MODELO_AUX,
              historial: list[tuple[str, str]] | None = None,
              condensar: bool = True) -> Recuperacion:
    """Pipeline completo de recuperación multi-consulta con fusión RRF."""
    tiempos: dict[str, float] = {}

    t = time.time()
    consulta = condensar_pregunta(cliente, pregunta, historial or [], modelo_aux) \
        if (condensar and historial) else pregunta
    tiempos["condensado"] = time.time() - t

    t = time.time()
    variantes = generar_variantes(cliente, consulta, n_variantes, modelo_aux, hyde=hyde)
    tiempos["variantes"] = time.time() - t

    t = time.time()
    vectores = cliente.embeber(variantes, consulta=True)
    rankings: list[list[tuple[int, float]]] = []
    procedencia: dict[int, list[str]] = {}

    for variante, vector in zip(variantes, vectores):
        denso = indice.buscar_denso(vector, k_candidatos)
        rankings.append(denso)
        for idx, _ in denso:
            procedencia.setdefault(idx, []).append(f"densa: {variante[:60]}")
        if hibrido:
            lexico = indice.buscar_lexico(variante, k_candidatos)
            rankings.append(lexico)
            for idx, _ in lexico:
                procedencia.setdefault(idx, []).append(f"BM25: {variante[:60]}")
    tiempos["busqueda"] = time.time() - t

    puntos = fusionar_rrf(rankings)
    candidatos = sorted(puntos, key=puntos.get, reverse=True)[:k_candidatos]

    t = time.time()
    if usar_rerank and len(candidatos) > k_final:
        finales = reordenar_llm(cliente, consulta, indice, candidatos, k_final, modelo_aux)
    elif usar_mmr:
        finales = mmr(indice, candidatos, vectores[0], k_final)
    else:
        finales = candidatos[:k_final]
    tiempos["reordenado"] = time.time() - t

    return Recuperacion(
        trozos=[indice.trozos[i] for i in finales],
        puntuaciones=[round(puntos.get(i, 0.0), 5) for i in finales],
        variantes=variantes,
        procedencia={indice.trozos[i].id: sorted(set(procedencia.get(i, []))) for i in finales},
        tiempos=tiempos,
        pregunta_usada=consulta,
    )


def construir_prompt(pregunta: str, trozos: Sequence[Trozo],
                     historial: list[tuple[str, str]] | None = None) -> str:
    contexto = "\n\n".join(
        f'<fragmento id="{t.id}" fuente="{t.doc}"'
        + (f' apartado="{t.seccion}"' if t.seccion else "") + f">\n{t.texto}\n</fragmento>"
        for t in trozos)
    previo = ""
    if historial:
        previo = "Conversación previa (contexto, no fuente de datos):\n" + "\n".join(
            f"Usuario: {u}\nAsistente: {a[:300]}" for u, a in historial[-2:]) + "\n\n"
    return f"{previo}Fragmentos recuperados:\n\n{contexto}\n\nPregunta: {pregunta}\n\nRespuesta:"


# --------------------------------------------------------------------------------------
# Lectura de ficheros
# --------------------------------------------------------------------------------------

EXTENSIONES = {".pdf", ".md", ".txt", ".markdown", ".csv", ".json"}


def leer_pdf(datos: bytes) -> str:
    from pypdf import PdfReader
    import io
    lector = PdfReader(io.BytesIO(datos))
    return "\n".join((p.extract_text() or "") for p in lector.pages)


def leer_fichero(nombre: str, datos: bytes) -> str:
    ext = Path(nombre).suffix.lower()
    if ext == ".pdf":
        return leer_pdf(datos)
    for cod in ("utf-8", "latin-1"):
        try:
            return datos.decode(cod)
        except UnicodeDecodeError:
            continue
    return datos.decode("utf-8", errors="ignore")


# --------------------------------------------------------------------------------------
# Pruebas
# --------------------------------------------------------------------------------------

def _selftest() -> None:
    """Valida troceado, BM25, RRF y MMR sin tocar la API."""
    print("== selftest (sin API) ==")
    texto = (
        "1. Introducción\nEste documento describe la plataforma de datos y sus servicios "
        "principales. El alcance es interno.\n\n"
        "2. Retención\nLos logs de auditoría se conservan 400 dias por requisito supervisor. "
        "La zona raw retiene 30 dias.\n\n"
        "3. Códigos de error\nEl código HLS-4093 indica cuota de peticiones agotada. "
        "El código HLS-4110 indica fallo de validación de token.\n\n"
        "4. Tarifas\nLa unidad de computo cuesta 0,042 euros. El almacenamiento cuesta "
        "21,50 euros por TB-mes.\n"
    ) * 3

    for est in ("ventana", "recursivo", "estructural"):
        trozos = trocear_documento("demo.md", texto, est, 400, 80)
        assert trozos, est
        assert all(len(t.texto) <= 900 for t in trozos), f"{est}: trozo demasiado grande"
        print(f"  {est:12s} -> {len(trozos):3d} trozos, "
              f"máx {max(len(t.texto) for t in trozos)} chars, "
              f"secciones: {sorted({t.seccion for t in trozos if t.seccion})[:2]}")

    # Índice con embeddings simulados (hash) para probar la mecánica.
    trozos = trocear_documento("demo.md", texto, "estructural", 400, 80)
    rng = np.random.default_rng(0)
    m = rng.normal(size=(len(trozos), 32)).astype("float32")
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    idx = Indice("demo", trozos, m, {})

    v = m[2]
    denso = idx.buscar_denso(v, 3)
    assert denso[0][0] == 2 and denso[0][1] > 0.99, "la búsqueda densa no recupera el propio vector"
    print(f"  denso        -> ok (top1={denso[0][0]}, sim={denso[0][1]:.3f})")

    lex = idx.buscar_lexico("HLS-4093", 3)
    mejor = idx.trozos[lex[0][0]].texto
    assert "HLS-4093" in mejor, "BM25 no encuentra el código exacto"
    print(f"  BM25         -> ok, top1 contiene HLS-4093")

    fus = fusionar_rrf([[(5, 0.9), (1, 0.8)], [(1, 3.2), (7, 1.1)]])
    assert max(fus, key=fus.get) == 1, "RRF debería premiar lo que aparece en ambos rankings"
    print(f"  RRF          -> ok (gana el idx 1, presente en los dos rankings)")

    sel = mmr(idx, list(range(len(trozos)))[:6], v, 3)
    assert len(sel) == 3 and len(set(sel)) == 3
    print(f"  MMR          -> ok ({sel})")

    assert normalizar("Auditoría ÁÉÍÓÚ") == "auditoria aeiou"
    assert "hls-4093" in tokenizar("El código HLS-4093 falla")
    print("  utilidades   -> ok")
    print("\nTodo correcto. Lanza la interfaz con:  streamlit run app.py")


def _smoke() -> None:
    """Prueba de humo contra la API real (consume cuota mínima)."""
    print("== smoke test (usa la API) ==")
    cli = ClienteGemini()
    gen, emb = cli.listar_modelos()
    print(f"  modelos de generación disponibles: {len(gen)} (p. ej. {gen[:3]})")
    print(f"  modelos de embedding disponibles : {emb}")
    v = cli.embeber(["hola mundo", "adiós mundo"])
    print(f"  embeddings -> matriz {v.shape}, norma {np.linalg.norm(v[0]):.3f}")
    print(f"  generación -> {cli.generar('Responde solo: OK', max_tokens=10)!r}")
    variantes = generar_variantes(cli, "¿Cuánto se conservan los logs de auditoría?", 3, MODELO_AUX)
    print("  variantes generadas:")
    for x in variantes:
        print("   ·", x)
    print(f"  tokens: entrada={cli.tokens_entrada} salida={cli.tokens_salida} "
          f"embedding≈{cli.tokens_embedding}")


if __name__ == "__main__":
    import sys
    if "--smoke" in sys.argv:
        _smoke()
    else:
        _selftest()
