from fastmcp import FastMCP
from datetime import datetime
from typing import Literal
from ddgs import DDGS

mcp = FastMCP(name="Demo de Servidor MCP con Herramientas Básicas")

@mcp.tool()
def saludar(nombre: str) -> str:
    """
    Saluda a una persona por su nombre.
    
    Args:
        nombre: El nombre de la persona a saludar
        
    Returns:
        Un mensaje de saludo personalizado
    """
    return f"¡Hola, {nombre}! Bienvenido al mundo de MCP. 🎉"


@mcp.tool()
def calcular(
    operacion: Literal["suma", "resta", "multiplicacion", "division"],
    a: float,
    b: float
) -> str:
    """
    Realiza operaciones matemáticas básicas.
    
    Args:
        operacion: Tipo de operación (suma, resta, multiplicacion, division)
        a: Primer número
        b: Segundo número
        
    Returns:
        El resultado de la operación
    """
    operaciones = {
        "suma": lambda x, y: x + y,
        "resta": lambda x, y: x - y,
        "multiplicacion": lambda x, y: x * y,
        "division": lambda x, y: x / y if y != 0 else "Error: división por cero"
    }
    
    if operacion not in operaciones:
        return f"Operación '{operacion}' no soportada"
    
    resultado = operaciones[operacion](a, b)
    simbolos = {"suma": "+", "resta": "-", "multiplicacion": "×", "division": "÷"}
    
    return f"{a} {simbolos[operacion]} {b} = {resultado}"


@mcp.tool()
def hora_actual() -> str:
    """
    Obtiene la fecha y hora actual del servidor.
    
    Returns:
        La fecha y hora actual formateada
    """
    ahora = datetime.now()
    return ahora.strftime("📅 %d/%m/%Y - 🕐 %H:%M:%S")


@mcp.tool()
def generar_lista(elementos: list[str], titulo: str = "Lista") -> str:
    """
    Genera una lista formateada en Markdown.
    
    Args:
        elementos: Lista de elementos a incluir
        titulo: Título de la lista (opcional)
        
    Returns:
        Lista formateada en Markdown
    """
    if not elementos:
        return "La lista está vacía"
    
    items = "\n".join([f"- {item}" for item in elementos])
    return f"**{titulo}**\n{items}"


@mcp.tool()
def buscar_palabra(texto: str, palabra: str) -> str:
    """
    Busca una palabra en un texto y devuelve información.
    
    Args:
        texto: El texto donde buscar
        palabra: La palabra a buscar
        
    Returns:
        Información sobre las ocurrencias encontradas
    """
    texto_lower = texto.lower()
    palabra_lower = palabra.lower()
    
    ocurrencias = texto_lower.count(palabra_lower)
    
    if ocurrencias == 0:
        return f"La palabra '{palabra}' no se encontró en el texto."
    
    return f"La palabra '{palabra}' aparece {ocurrencias} vez(es) en el texto."


@mcp.tool()
def convertir_temperatura(
    valor: float,
    de: Literal["celsius", "fahrenheit", "kelvin"],
    a: Literal["celsius", "fahrenheit", "kelvin"]
) -> str:
    """
    Convierte temperaturas entre diferentes unidades.
    
    Args:
        valor: El valor de temperatura a convertir
        de: Unidad de origen
        a: Unidad de destino
        
    Returns:
        La temperatura convertida
    """
    # Primero convertir a Celsius
    if de == "fahrenheit":
        celsius = (valor - 32) * 5/9
    elif de == "kelvin":
        celsius = valor - 273.15
    else:
        celsius = valor
    
    # Luego convertir de Celsius a la unidad destino
    if a == "fahrenheit":
        resultado = celsius * 9/5 + 32
    elif a == "kelvin":
        resultado = celsius + 273.15
    else:
        resultado = celsius
    
    simbolos = {"celsius": "°C", "fahrenheit": "°F", "kelvin": "K"}
    return f"{valor} {simbolos[de]} = {resultado:.2f} {simbolos[a]}"


@mcp.tool()
def buscar_en_internet(consulta: str, max_resultados: int = 5) -> str:
    """
    Busca información en Internet usando DuckDuckGo.
    
    Args:
        consulta: La búsqueda a realizar
        max_resultados: Número máximo de resultados (por defecto 5)
        
    Returns:
        Los resultados de la búsqueda formateados
    """
    try:
        with DDGS() as ddgs:
            resultados = list(ddgs.text(consulta, max_results=max_resultados))
        
        if not resultados:
            return f"No se encontraron resultados para: '{consulta}'"
        
        texto = f"🔍 **Resultados para: '{consulta}'**\n\n"
        for i, r in enumerate(resultados, 1):
            titulo = r.get("title", "Sin título")
            url = r.get("href", "")
            descripcion = r.get("body", "Sin descripción")
            texto += f"**{i}. {titulo}**\n"
            texto += f"   🔗 {url}\n"
            texto += f"   {descripcion}\n\n"
        
        return texto
    except Exception as e:
        return f"Error al buscar: {str(e)}"


@mcp.prompt()
def bienvenida() -> str:
    """
    Prompt de bienvenida que se muestra al iniciar el servidor.
    
    Returns:
        Un mensaje de bienvenida para los usuarios
    """
    return "¡Bienvenido al Servidor MCP Demo! 🚀 Aquí puedes probar varias herramientas útiles. ¡Explora y diviértete!"

@mcp.resource("info://servidor")
def info_servidor() -> str:
    """Devuelve información sobre el servidor."""
    return "Servidor MCP Demo v1.0 - Ejecutándose correctamente"

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)