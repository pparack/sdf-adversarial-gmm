import os
import numpy as np
import torch
from .models import SDFNet, CriticNet
from .eval import eval_summary_sdf_only
from .utils_seed import set_seed

def grad_penalty_x_only(critic, x, y, lambda_gp=10.0):
    x = x.requires_grad_(True)
    out = critic(x, y)
    ones = torch.ones_like(out)
    grad_x, = torch.autograd.grad(
        outputs=out, inputs=x, grad_outputs=ones,
        create_graph=True, retain_graph=True, only_inputs=True
    )
    gp = ((grad_x.view(x.size(0), -1).norm(2, dim=1) - 1.0) ** 2).mean() * lambda_gp
    return gp

def train_adv_once(
    input_dim,
    train_loader, valid_loader, test_loader,
    device,
    lambda_gp=10.0,
    gamma_baseline=4.0,
    critic_steps=3,
    alpha_mean=10.0,
    beta_std=1.0,
    max_epochs_adv=40,
    patience_adv=12,
    hidden_sdf=20,
    hidden_crit=32,
    dropout_sdf=0.2,
    dropout_crit=0.2,
    use_spectral_norm=True,
    lr_sdf=5e-4,
    lr_crit=8e-4,
    wd_sdf=1e-4,
    wd_crit=5e-4,
    seeds=(41,42,43),
    tag="exp",
    save_dir="results"
):
    os.makedirs(save_dir, exist_ok=True)
    valid_hist, test_hist = [], []
    best_val_over_seeds = float("inf")
    best_model_path = None

    for seed in seeds:
        set_seed(seed)
        sdf_adv = SDFNet(input_dim, hidden=hidden_sdf, drop=dropout_sdf).to(device)
        critic  = CriticNet(input_dim, hidden=hidden_crit, dropout=dropout_crit, use_spectral_norm=use_spectral_norm).to(device)
        opt_sdf    = torch.optim.Adam(sdf_adv.parameters(), lr=lr_sdf,  weight_decay=wd_sdf)
        opt_critic = torch.optim.Adam(critic.parameters(), lr=lr_crit, weight_decay=wd_crit)

        # warm-up
        for _ in range(3):
            sdf_adv.train()
            for X, y in train_loader:
                opt_sdf.zero_grad()
                m = sdf_adv(X).squeeze(1)
                loss = (m*y).pow(2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sdf_adv.parameters(), 0.5)
                opt_sdf.step()

        best_val, wait = float("inf"), 0
        model_path = os.path.join(save_dir, f"sdf_adv_{tag}_seed{seed}.pt")

        for epoch in range(1, max_epochs_adv+1):
            sdf_adv.train(); critic.train()
            for X, y in train_loader:
                # critic K steps
                for _ in range(critic_steps):
                    opt_critic.zero_grad()
                    with torch.no_grad():
                        m = sdf_adv(X).detach().squeeze(1)
                    g = critic(X, y).squeeze(1)
                    loss_c = -(m*y*g).mean() + grad_penalty_x_only(critic, X, y, lambda_gp=lambda_gp)
                    loss_c.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    opt_critic.step()

                # sdf 1 step
                opt_sdf.zero_grad()
                m = sdf_adv(X).squeeze(1)
                with torch.no_grad():
                    g = critic(X, y).squeeze(1)
                moment_g = m*y*g
                loss_s = (moment_g**2).mean()                        + gamma_baseline * (m*y).pow(2).mean()                        + alpha_mean * (m.mean()-1.0).pow(2)                        + beta_std  * (m.std(unbiased=False)-1.0).pow(2)
                loss_s.backward()
                torch.nn.utils.clip_grad_norm_(sdf_adv.parameters(), 0.5)
                opt_sdf.step()

            val_score = eval_summary_sdf_only(sdf_adv, valid_loader, device=device)["Euler_MSE"]
            if epoch == 1 or val_score < best_val - 1e-9:
                best_val, wait = val_score, 0
                torch.save(sdf_adv.state_dict(), model_path)
            else:
                wait += 1
                if wait >= patience_adv:
                    break

        sdf_adv.load_state_dict(torch.load(model_path, map_location=device))
        v = eval_summary_sdf_only(sdf_adv, valid_loader, device=device)
        t = eval_summary_sdf_only(sdf_adv, test_loader, device=device)
        valid_hist.append(v); test_hist.append(t)

        if v["Euler_MSE"] < best_val_over_seeds:
            best_val_over_seeds = v["Euler_MSE"]
            best_model_path = model_path

    def _agg(hist, key):
        arr = np.array([h[key] for h in hist], dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0))

    out = {
        "tag": tag,
        "lambda_gp": lambda_gp,
        "gamma_baseline": gamma_baseline,
        "critic_steps": critic_steps,
        "valid_Euler_MSE_mean": _agg(valid_hist, "Euler_MSE")[0],
        "valid_Euler_MSE_std":  _agg(valid_hist, "Euler_MSE")[1],
        "test_Euler_MSE_mean":  _agg(test_hist,  "Euler_MSE")[0],
        "test_Euler_MSE_std":   _agg(test_hist,  "Euler_MSE")[1],
        "best_model_path": best_model_path
    }
    return out
