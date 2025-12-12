import os
import numpy as np
import torch
from .models import SDFNet
from .eval import eval_summary_sdf_only
from .utils_seed import set_seed

def train_soft_baseline(
    input_dim,
    train_loader, valid_loader, test_loader,
    device,
    seed=42,
    hidden=20, drop=0.2,
    lr=1e-3, weight_decay=1e-4,
    grad_clip=1.0,
    max_epochs=100, patience=10,
    save_path=None
):
    set_seed(seed)
    model = SDFNet(input_dim, hidden=hidden, drop=drop).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if save_path is None:
        save_path = f"results/baseline_sdf_seed{seed}.pt"

    best_val, wait = float("inf"), 0
    for epoch in range(1, max_epochs+1):
        model.train()
        for X, y in train_loader:
            opt.zero_grad()
            m = model(X).squeeze(1)
            loss = (m*y).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        val = eval_summary_sdf_only(model, valid_loader, device=device)["Euler_MSE"]
        if epoch == 1 or val < best_val - 1e-9:
            best_val, wait = val, 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(torch.load(save_path, map_location=device))
    return {
        "seed": seed,
        "valid": eval_summary_sdf_only(model, valid_loader, device=device),
        "test":  eval_summary_sdf_only(model, test_loader,  device=device),
        "path":  save_path
    }
