import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from prott3_p2t.config import Config
from prott3_p2t.data import load_and_prepare_data, login_huggingface
from prott3_p2t.dataset import create_dataloaders
from prott3_p2t.esm_cache import load_esm, precompute_esm_cache
from prott3_p2t.evaluate import (
    compute_bertscore,
    compute_bleu_rouge,
    generate_predictions,
    print_examples,
)
from prott3_p2t.train import (
    build_model_from_loader,
    build_optimizer_and_scheduler,
    save_model,
    train_model,
)


def main():
    config = Config()
    config.ensure_dirs()

    login_huggingface()

    train_data_df, val_data_df = load_and_prepare_data(config)

    esm_tokenizer, esm_model = load_esm(config)
    precompute_esm_cache(
        train_data_df,
        config=config,
        esm_tokenizer=esm_tokenizer,
        esm_model=esm_model,
        batch_size=config.esm_batch_size,
    )
    precompute_esm_cache(
        val_data_df,
        config=config,
        esm_tokenizer=esm_tokenizer,
        esm_model=esm_model,
        batch_size=config.esm_batch_size,
    )

    train_loader, val_loader = create_dataloaders(train_data_df, val_data_df, config)
    print(len(train_loader))
    print(len(val_loader))

    model = build_model_from_loader(train_loader, config)
    optimizer, scheduler = build_optimizer_and_scheduler(model, train_loader, config)

    model = train_model(model, train_loader, optimizer, scheduler, config)
    save_model(model, config)

    df_results = generate_predictions(model, val_data_df, config)
    df_results.to_csv("predictions.csv", index=False)

    compute_bleu_rouge(df_results)
    compute_bertscore(df_results)
    print_examples(df_results, n=config.num_examples_to_print)


if __name__ == "__main__":
    main()
