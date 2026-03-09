# GIL-Exist-2026

OCR de memes usando DeepSeek-OCR

Este proyecto implementa un pipeline para extraer texto de memes mediante OCR utilizando un modelo multimodal basado en transformers. El sistema procesa imágenes almacenadas en una carpeta, ejecuta el modelo de OCR y guarda el texto extraído en un archivo Excel para su posterior análisis.

El objetivo es automatizar la extracción de texto embebido en memes, lo cual permite utilizar posteriormente el contenido textual en tareas de procesamiento de lenguaje natural como clasificación, análisis de discurso o detección de contenido ofensivo.

Modelo utilizado

El modelo utilizado fue:

DeepSeek-OCR

Repositorio base del modelo:

https://huggingface.co/deepseek-ai/DeepSeek-OCR

Para optimizar el uso de memoria GPU se utilizó la versión cuantizada en 4 bits:

Jalea96/DeepSeek-OCR-bnb-4bit-NF4

https://huggingface.co/Jalea96/DeepSeek-OCR-bnb-4bit-NF4

Características del modelo

Arquitectura multimodal basada en Vision-Language Transformers

Optimizado para reconocimiento óptico de caracteres (OCR)

Cuantización 4-bit NF4 mediante bitsandbytes

Uso eficiente de memoria GPU

Compatible con PyTorch y HuggingFace Transformers

La versión cuantizada permite ejecutar el modelo en GPUs de consumo como la RTX 4060 manteniendo un alto rendimiento en la extracción de texto.
