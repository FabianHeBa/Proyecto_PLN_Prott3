# Prott3 Protein2Text-QA

Este proyecto implementa una estructura modular para experimentar con un sistema de generación de respuestas en el dominio biomédico a partir de secuencias de proteínas. La idea general es combinar representaciones biológicas obtenidas con un encoder especializado en proteínas con la capacidad generativa de un modelo de lenguaje, de forma que el sistema pueda responder preguntas asociadas a información proteica.

La arquitectura sigue un flujo inspirado en Prott3, donde primero se extraen embeddings de las secuencias mediante ESM-2. Después, estas representaciones se adaptan al espacio del modelo de lenguaje usando un Q-Former, que funciona como puente entre la información proteica y el generador textual. Finalmente, Gemma se ajusta mediante LoRA para producir respuestas condicionadas tanto por la pregunta como por la representación aprendida de la proteína.

```text
ESM-2 -> Q-Former -> Gemma + LoRA
```
El dataset utilizado es `tumorailab/Protein2Text-QA`, que contiene pares de pregunta-respuesta asociados a secuencias de proteínas. Durante el flujo se precomputan los embeddings ESM para reducir el costo durante el entrenamiento, se entrena el Q-Former junto con adaptadores LoRA sobre Gemma y se evalúa el desempeño del modelo mediante métricas de generación de texto como BLEU, ROUGE y BERTScore.


## Estructura

```text
prott3_p2t_github/
├── prott3_p2t/
│   ├── config.py        # Constantes e hiperparámetros
│   ├── data.py          # Carga del dataset y extracción pregunta/respuesta
│   ├── esm_cache.py     # Precomputo de embeddings ESM-2
│   ├── dataset.py       # Dataset, collator y DataLoaders
│   ├── qformer.py       # ProteinQFormer
│   ├── model.py         # ProtT3GemmaQA
│   ├── train.py         # Optimizer, scheduler, loop de entrenamiento y guardado
│   └── evaluate.py      # Generación de predicciones y métricas
├── scripts/
│   ├── run_pipeline.py  # Ejecuta todo el flujo original
│   └── precompute_esm.py
├── requirements.txt
├── .env.example
└── .gitignore
```

## Versiones principales

El proyecto está preparado para las versiones que se usaron en la workstation:

```text
transformers==4.44.0
accelerate==0.34.0
peft==0.12.0
bitsandbytes==0.45.3
```

## Instalación

```bash
pip install -r requirements.txt
```

Si el modelo o dataset requiere autenticación, configura el token de Hugging Face como variable de entorno:

```bash
export HF_TOKEN="tu_token"
```

En Windows PowerShell:

```powershell
$env:HF_TOKEN="tu_token"
```

## Ejecución

Desde la raíz del repositorio:

```bash
python scripts/run_pipeline.py
```

El script hace lo mismo que el notebook original:

1. Carga `tumorailab/Protein2Text-QA`.
2. Toma el 30% del split `test`.
3. Divide en train/validation con `test_size=0.05`.
4. Extrae columnas `question` y `answer` desde `conversations`.
5. Precomputa embeddings ESM-2 en `./esm_cache`.
6. Entrena `ProtT3GemmaQA`.
7. Guarda el modelo en `./prott3_gemma_qa`.
8. Genera predicciones en validación.
9. Calcula BLEU, ROUGE y BERTScore.

## Salidas ignoradas por Git

Los siguientes artefactos se generan localmente y no deberían subirse al repositorio:

```text
esm_cache/
prott3_gemma_qa/
predictions.csv
*.pt
*.safetensors
```
