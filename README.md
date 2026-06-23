# Clasificador de Contenidos Audiovisuales con Modelos de Lenguaje Multimodales (MLLM)

Este proyecto implementa un pipeline avanzado de clasificación de contenido televisivo basado en el estándar DVB (ETSI EN 300 468). Utiliza modelos de lenguaje visuales (MLLM) para analizar metadatos EPG extraídos de la tabla EIT y frames de vídeo, asignando cada programa a una de las 10 categorías oficiales.

Además, cuenta con un sistema de monitorización de hardware que registra latencias, consumo energético (GPU/CPU a través de NVML y RAPL) y uso de memoria durante la inferencia.

---

## Características del Proyecto

Sobre este proyecto se pueden realizar diferentes pruebas, en función del tipo de entrada dada al modelo.

- **Clasificación Textual**: Soporta análisis de texto puro (metadatos EIT)
- **Clasificación Visual**: Mediante la extracción temporal de frames.
- **Clasificación Multimodal**: Aproximación híbrida de imagen + texto.

- **Pipeline de Dos Fases (Two-Stage)**:

   - **visual_then_epg**: Votación visual seguida de una resolución final usando EPG.
   - **epg_then_visual**: La EPG filtra el Top 3 de categorías, dejando la decisión final entre este top 3 al análisis visual.


**Monitorización de Recursos**: Captura uso de GPU, VRAM, RAM, consumo en Julios (energía) y vatios (potencia), y velocidad de generación (tokens/s).

---

##  Estructura de Directorios

La estructura del código debe ser la siguiente:

<img width="703" height="208" alt="estructura_archivos" src="https://github.com/user-attachments/assets/51a78ca5-217f-4324-93e8-ae6ed5521c3e" />

--- 

##  Comando de ejecución

python3 automatizacion.py


# Clasificador de Contenidos Audiovisuales DVB con Modelos de Lenguaje Multimodales (MLLM)

Este proyecto es una herramienta avanzada en Python diseñada para implementar un pipeline de clasificación de contenido televisivo basándose en el estándar DVB (ETSI EN 300 468). Utiliza Modelos de Lenguaje Visuales (MLLM) integrados con HuggingFace para analizar metadatos EPG (extraídos de tablas EIT) y *frames* de vídeo, asignando automáticamente cada programa a una de las 10 categorías oficiales.

## Características Principales

* **Clasificación Flexible y Multimodal:** Soporta análisis de texto puro mediante la ingesta de archivos `.eit.xml`, análisis puramente visual mediante muestreo de *frames*, o una aproximación híbrida (Multimodal) que combina el contexto de la EPG con la imagen.
* **Pipeline Inteligente de Dos Fases (Two-Stage):** Emplea lógicas de resolución complejas para maximizar el acierto:
    * *Visual_then_EPG:* Realiza una votación visual frame a frame y delega la resolución final a la EPG.
    * *EPG_then_Visual:* La EPG filtra un Top 3 de categorías candidatas, dejando la decisión definitiva al análisis de las imágenes.
* **Monitorización Exhaustiva de Hardware:** Registra el impacto en el sistema durante la inferencia capturando el consumo energético en Julios y Vatios (mediante `NVML` y `pyRAPL`), uso de memoria RAM/VRAM, y velocidades de generación (tokens/s).
* **Reportes Detallados:** Genera de forma automatizada archivos JSON por programa y un registro global en formato CSV, contrastando las predicciones del modelo con el *Ground Truth* manual y el de la EPG.

## Arquitectura y Módulos del Sistema

El flujo de trabajo se coordina mediante distintos scripts interconectados:

* `pipeline.py`: Contiene el núcleo de inferencia. Construye el *prompt* de contexto cruzando los datos DVB, se comunica con el MLLM y parsea la respuesta JSON garantizando que se cumplan las categorías estándar.
* `metrics_monitor.py`: Hilo paralelo en el entorno que levanta los sensores del hardware para monitorizar de manera pasiva el comportamiento del proceso sin bloquear la inferencia.
* `run_all_videos.py`: Orquestador principal que mapea la jerarquía de directorios, carga las etiquetas base (`dataset_definitivo.csv`) y vuelca los resultados de la métrica en CSV/JSON.
* `automatizacion.py`: Módulo pensado para investigaciones, permite lanzar secuencialmente un bloque predefinido de experimentos (stride de 30s vs 60s, solo texto, solo frames, etc.).

### Estructura de Directorios

El código espera que los datos de entrada mantengan la siguiente estructura:

<img width="703" height="208" alt="estructura_archivos" src="https://github.com/user-attachments/assets/51a78ca5-217f-4324-93e8-ae6ed5521c3e" />

## Uso

El script principal se ejecuta a través de la terminal, pudiendo escoger el modo de prueba automatizado o la ejecución parametrizada.

**1. Ejecutar la Batería de Experimentos Automatizados**
Esta es la opción recomendada para validar todos los enfoques de una vez. Generará carpetas separadas para resultados de solo-texto, solo-frames, y *two-stage*:
```bash
python3 automatizacion.py
