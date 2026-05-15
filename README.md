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

## Prerrequisitos

- **Docker** instalado en tu sistema. 
- **NVIDIA Container Toolkit** configurado para permitir a Docker acceder a la GPU.

##  Estructura de Directorios

La estructura del código debe ser la siguiente:

📁 codigos_definitivos_MLLM/
├── 📄 .env                       # Variables de entorno (token hugging face)
├── 📄 dataset_definitivo.csv     # Archivo CSV de entrada con la información de los vídeos
├── 📁 carpeta_frames_completa/   # Directorio con los frames extraídos de los vídeos
├── 📁 eit/                       # Directorio de metadatos EIT
├── 📁 resultados/                # (Se autocompleta) Aquí se guardarán los JSON y el CSV final
└── 📁 logs/                      # (Se autocompleta) Archivos de registro y monitorización

--- 

##  Comando de ejecución

python3 automatizacion.py
