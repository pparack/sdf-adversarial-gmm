import torch
from pathlib import Path

from src.config import default_paths
from src.preprocess import preprocess_openap, check_splits
from src.dataset import make_tensors_and_loaders
from src.train_soft import train_soft_baseline
from src.train_adv import train_adv_once

def main():
    paths = default_paths()
    data_path = paths.data_dir / "OpenAP_Macro.parquet.gzip"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Put your parquet here: {data_path}\n"
            "NOTE: data/ is gitignored, so it stays local."
        )

    split = preprocess_openap(str(data_path))
    check_splits(split.train_df, split.valid_df, split.test_df,
                 split.feature_names, split.macro_cols)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (train_X_t, train_y_t, valid_X_t, valid_y_t, test_X_t, test_y_t,
     train_loader, valid_loader, test_loader) = make_tensors_and_loaders(
        split.train_df, split.valid_df, split.test_df, split.feature_names, device=device
    )
    input_dim = train_X_t.shape[1]

    soft = train_soft_baseline(
        input_dim=input_dim,
        train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
        device=device,
        seed=42
    )
    print("SOFT:", soft["valid"], soft["test"])

    adv = train_adv_once(
        input_dim=input_dim,
        train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
        device=device,
        lambda_gp=12.0, gamma_baseline=4.0, critic_steps=2,
        seeds=(41,42,43),
        tag="p2_lgp12_gb4_cs2",
        save_dir=str(paths.results_dir)
    )
    print("ADV:", adv)

if __name__ == "__main__":
    main()
