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

<img width="557" height="116" alt="estructura_archivos" src="https://github.com/user-attachments/assets/882750c9-668d-4d40-ae21-a414199ee978" />


--- 

##  Comando de ejecución

python3 automatizacion.py
