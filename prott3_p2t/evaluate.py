import pandas as pd
import torch
from tqdm import tqdm

from .config import Config
from .esm_cache import cache_path


def generate_predictions(model, val_data_df, config: Config) -> pd.DataFrame:
    model.eval()

    predictions = []
    ground_truths = []
    questions = []

    for _, row in tqdm(
        val_data_df.iterrows(),
        total=len(val_data_df),
        desc="Generando predicciones",
    ):
        question = str(row[config.question_col]).strip()
        answer = str(row[config.answer_col]).strip()

        protein_emb = torch.load(
            cache_path(row[config.seq_col], config.cache_dir),
            map_location="cpu",
        ).unsqueeze(0).float()
        protein_mask = torch.ones(1, protein_emb.shape[1], dtype=torch.long)

        pred_text = model.generate(protein_emb, protein_mask, question)

        predictions.append(pred_text)
        ground_truths.append(answer)
        questions.append(row[config.question_col])

    df_results = pd.DataFrame(
        {
            "prediction": predictions,
            "ground_truth": ground_truths,
            "question": questions,
        }
    )

    return df_results


def compute_bleu_rouge(df_results: pd.DataFrame):
    import evaluate

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    preds = df_results["prediction"].tolist()
    refs_bleu = [[gt] for gt in df_results["ground_truth"].tolist()]
    refs_rouge = df_results["ground_truth"].tolist()

    bleu_output = bleu.compute(predictions=preds, references=refs_bleu)
    rouge_output = rouge.compute(predictions=preds, references=refs_rouge)

    print("--- RESULTADOS DE EVALUACIÓN ---")
    print(f"BLEU Score: {bleu_output['bleu']:.4f}")
    print(f"ROUGE-1: {rouge_output['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_output['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_output['rougeL']:.4f}")

    return bleu_output, rouge_output


def compute_bertscore(df_results: pd.DataFrame):
    import evaluate

    bertscore = evaluate.load("bertscore")

    preds = df_results["prediction"].tolist()
    refs_rouge = df_results["ground_truth"].tolist()

    bertscore_output = bertscore.compute(
        predictions=preds,
        references=refs_rouge,
        lang="en",
        model_type="distilbert-base-uncased",
    )

    print(
        f"BERTScore Precision: "
        f"{sum(bertscore_output['precision']) / len(bertscore_output['precision']):.4f}"
    )
    print(
        f"BERTScore Recall:    "
        f"{sum(bertscore_output['recall']) / len(bertscore_output['recall']):.4f}"
    )
    print(
        f"BERTScore F1:        "
        f"{sum(bertscore_output['f1']) / len(bertscore_output['f1']):.4f}"
    )

    return bertscore_output


def print_examples(df_results: pd.DataFrame, n: int = 10):
    for i in range(min(n, len(df_results))):
        row = df_results.iloc[i]
        prediccion = row["prediction"]
        verdad = row["ground_truth"]
        pregunta = row["question"]
        print(f"id{i}")
        print(f"Pregunta: {pregunta}")
        print("\n")
        print(f"Predcción:{prediccion}")
        print("\n")
        print(f"Verdad: {verdad}")
        print("-" * 84)
