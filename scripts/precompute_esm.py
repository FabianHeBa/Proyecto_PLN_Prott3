import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from prott3_p2t.config import Config
from prott3_p2t.data import load_and_prepare_data, login_huggingface
from prott3_p2t.esm_cache import load_esm, precompute_esm_cache


def main():
    config = Config()
    config.ensure_dirs()

    login_huggingface()
    train_data_df, val_data_df = load_and_prepare_data(config)

    esm_tokenizer, esm_model = load_esm(config)
    precompute_esm_cache(train_data_df, config, esm_tokenizer, esm_model)
    precompute_esm_cache(val_data_df, config, esm_tokenizer, esm_model)


if __name__ == "__main__":
    main()
