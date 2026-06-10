# Prott3 Protein2Text-QA

Este proyecto toma como punto de partida la arquitectura propuesta en el siguiente artículo:

Liu, Zhiyuan, An Zhang, Hao Fei, Enzhi Zhang, Xiang Wang, Kenji Kawaguchi, and Tat-Seng Chua. 2024. ProtT3: Protein-to-Text Generation for Text-based Protein Understanding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5949–5966, Bangkok, Thailand. Association for Computational Linguistics. DOI: 10.18653/v1/2024.acl-long.324.

```bibtex
@inproceedings{liu-etal-2024-prott3,
    title = "{P}rot{T}3: Protein-to-Text Generation for Text-based Protein Understanding",
    author = "Liu, Zhiyuan and Zhang, An and Fei, Hao and Zhang, Enzhi and Wang, Xiang and Kawaguchi, Kenji and Chua, Tat-Seng",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.324/",
    doi = "10.18653/v1/2024.acl-long.324",
    pages = "5949--5966"
}
```

# Metodología
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
│   ├── run_pipeline.py  # Ejecuta todo el flujo
│   └── precompute_esm.py
├── requirements.txt
├── .env.example
└── .gitignore
```

## Versiones principales

Se usaron las siguientes versiones de estas librerias por limitaciones de hardware y compatibilidad 

```text
transformers==4.44.0
accelerate==0.34.0
peft==0.12.0
bitsandbytes==0.45.3
```

Se requiere autenticación, se puede configurar el token de Hugging Face como variable de entorno:

```bash
export HF_TOKEN="tu_token"
```

En Windows PowerShell:

```powershell
$env:HF_TOKEN="tu_token"
```

## Ejecución

El script `run_pipeline.py` ejecuta todo el flujo en el siguiente orden:

1. Carga `tumorailab/Protein2Text-QA`.
2. Toma el 30% del split `test`.
3. Divide en train/validation con `test_size=0.05`.
4. Extrae columnas `question` y `answer` desde `conversations`.
5. Precomputa embeddings ESM-2 en `./esm_cache`.
6. Entrena `ProtT3GemmaQA`.
7. Guarda el modelo en `./prott3_gemma_qa`.
8. Genera predicciones en validación.
9. Calcula BLEU, ROUGE y BERTScore.

## Salidas

El script genera los siguientes archivos y carpetas de manera local:

```text
esm_cache/
prott3_gemma_qa/
predictions.csv
*.pt
*.safetensors
```
