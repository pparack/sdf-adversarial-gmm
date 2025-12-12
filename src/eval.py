import numpy as np
import torch

@torch.no_grad()
def eval_summary_sdf_only(model, loader, device):
    model.eval()
    euler_mse, abs_mean, cnt = 0.0, 0.0, 0
    m_vals = []
    for X, y in loader:
        X = X.to(device); y = y.to(device)
        m = model(X).squeeze(1)
        moment = m * y
        euler_mse += torch.sum(moment**2).item()
        abs_mean  += torch.sum(torch.abs(moment)).item()
        cnt       += X.size(0)
        m_vals.append(m.detach().cpu())
    euler_mse /= cnt
    abs_mean  /= cnt
    m_all = torch.cat(m_vals).numpy()
    return {"Euler_MSE": euler_mse, "AbsMeanMoment": abs_mean, "Std(m)": float(np.std(m_all, ddof=0))}
