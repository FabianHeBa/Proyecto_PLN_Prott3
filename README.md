# Prott3 Protein2Text-QA

Implementación modular del notebook original para entrenar una arquitectura tipo:

```text
ESM-2 -> Q-Former -> Gemma + LoRA
```

La lógica del notebook se mantuvo: se usa `tumorailab/Protein2Text-QA`, se precomputan embeddings ESM, se entrena el Q-Former junto con LoRA sobre Gemma y se evalúa con BLEU, ROUGE y BERTScore.

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
