"""
Test trained MNIST models with different robustness loss weights lambda.

lambda is the robustness loss weight:
  total_loss = classification_loss + lambda * robustness_loss

This script uses alpha-CROWN optimized bounds for accurate verification.
"""

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.nn import CrossEntropyLoss
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import models
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--eps", type=float, default=0.1)
parser.add_argument("--norm", type=float, default=np.inf)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--num_images", type=int, default=500)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--optimize", action="store_true",
                    help="Use alpha-CROWN optimization for tighter bounds (slower but more accurate)")
args = parser.parse_args()

device = args.device if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Settings: eps={args.eps}, num_images={args.num_images}, "
      f"batch_size={args.batch_size}, optimize={args.optimize}\n")

model_configs = [
    ("cnn7-IBP",         "mnist_cnn7_ibp_eps01.pth",   models.cnn_7layer),
    ("cnn4-IBP",         "mnist_cnn_ibp.pth",          models.cnn_4layer),
    ("cnn4-CROWN",       "mnist_cnn_crown.pth",        models.cnn_4layer),
    ("cnn4-CROWN-IBP",   "mnist_cnn_crown_ibp.pth",    models.cnn_4layer),
]

lambda_values = [0.1, 0.5, 1.0]

transform = transforms.ToTensor()
testset = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
num_class = 10


def test_model(model, bm, loader, eps, num_images):
    """
    Test model and return metrics for all lambda values.

    Returns:
      results: {lam: {"verified": int, "clean_ce": float, "robust_ce": float, "total_loss": float}}
      total: int
    """
    results = {}
    for lam in lambda_values:
        results[lam] = {"verified": 0, "clean_ce_sum": 0.0, "robust_ce_sum": 0.0}

    total = 0

    with torch.no_grad():
        for data, labels in loader:
            if num_images > 0 and total >= num_images:
                break
            remain = num_images - total if num_images > 0 else data.size(0)
            data, labels = data[:remain].to(device), labels[:remain].to(device)
            bs = data.size(0)
            total += bs

            # Standard forward pass for clean accuracy and CE loss
            output = model(data)
            clean_ce = CrossEntropyLoss(reduction='sum')(output, labels)
            clean_correct = (output.argmax(1) == labels).sum().item()

            # Robustness verification bounds
            c = torch.eye(num_class, device=device)[labels].unsqueeze(1) \
                - torch.eye(num_class, device=device).unsqueeze(0)
            I = ~(labels.unsqueeze(1) == torch.arange(num_class, device=device).unsqueeze(0))
            c = c[I].view(bs, num_class - 1, num_class)

            data_ub = torch.clamp(data + eps, 0.0, 1.0)
            data_lb = torch.clamp(data - eps, 0.0, 1.0)
            ptb = PerturbationLpNorm(norm=args.norm, eps=eps, x_L=data_lb, x_U=data_ub)
            _ = bm(BoundedTensor(data, ptb))

            if args.optimize:
                # Use alpha-CROWN optimization for tighter bounds
                bm.set_bound_opts({
                    'optimize_bound_args': {
                        'iteration': 50,
                        'lr_alpha': 0.1,
                        'verbose': False,
                    }
                })
                lb, _ = bm.compute_bounds(
                    IBP=False, C=c, method='CROWN-Optimized',
                    bound_upper=False
                )
            else:
                # Standard CROWN-IBP mix (same as training)
                ilb, _ = bm.compute_bounds(IBP=True, C=c, method=None)
                clb, _ = bm.compute_bounds(IBP=False, C=c, method='backward', bound_upper=False)
                # Use CROWN-heavy mix for best results (like final training stage)
                lb = 0.1 * clb + 0.9 * ilb

            # Verified: all margin lower bounds > 0
            verified = (lb > 0).all(dim=1).sum().item()

            # Robust cross-entropy loss
            lb_padded = torch.cat((
                torch.zeros(size=(lb.size(0), 1), dtype=lb.dtype, device=lb.device),
                lb
            ), dim=1)
            fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
            robust_ce = CrossEntropyLoss(reduction='sum')(-lb_padded, fake_labels)

            for lam in lambda_values:
                results[lam]["verified"] += verified
                results[lam]["clean_ce_sum"] += clean_ce.item()
                results[lam]["robust_ce_sum"] += robust_ce.item()

    return results, total


# ========== MAIN ==========
print(f"{'='*110}")
print(f"{'Model':<20} {'Lambda':<8} {'Clean Acc':<12} {'Verified':<12} {'Total':<8} "
      f"{'Clean CE':<12} {'Robust CE':<12} {'Total Loss':<12} {'Time(s)':<8}")
print(f"{'='*110}")

all_results = {}

for name, file_path, model_fn in model_configs:
    print(f"\n--- {name} ({file_path}) ---")

    sd = torch.load(file_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model = model_fn(in_ch=1, in_dim=28)
    model.load_state_dict(sd)
    model.eval().to(device)

    dummy = torch.randn(1, 1, 28, 28).to(device)
    bm = BoundedModule(model, dummy, device=device, bound_opts={"conv_mode": "patches"})

    start = time.time()
    results, total = test_model(model, bm, loader, args.eps, args.num_images)
    elapsed = time.time() - start

    # Clean accuracy from first batch's results
    clean_correct = 0
    with torch.no_grad():
        eval_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
        t = 0
        for data, labels in eval_loader:
            if args.num_images > 0 and t >= args.num_images: break
            remain = args.num_images - t if args.num_images > 0 else data.size(0)
            data, labels = data[:remain].to(device), labels[:remain].to(device)
            t += data.size(0)
            out = model(data)
            clean_correct += (out.argmax(1) == labels).sum().item()
    clean_acc = 100.0 * clean_correct / total

    for lam in lambda_values:
        r = results[lam]
        verified_rate = 100.0 * r["verified"] / total
        avg_clean_ce = r["clean_ce_sum"] / total
        avg_robust_ce = r["robust_ce_sum"] / total
        avg_total_loss = avg_clean_ce + lam * avg_robust_ce
        print(f"  {name:<18} {lam:<8.1f} {clean_acc:<12.2f} {r['verified']:<12} "
              f"{total:<8} {avg_clean_ce:<12.4f} {avg_robust_ce:<12.4f} "
              f"{avg_total_loss:<12.4f} {elapsed:<8.1f}")

    all_results[name] = {"results": results, "total": total, "clean_acc": clean_acc}

print(f"\n\n{'='*110}")
print(f"{'Model':<20} {'Clean Acc':<10} {'λ=0.1 Verified':<16} "
      f"{'λ=0.5 Verified':<16} {'λ=1.0 Verified':<16}")
print(f"{'='*110}")
for name in all_results:
    t = all_results[name]["total"]
    r = all_results[name]["results"]
    ca = all_results[name]["clean_acc"]
    print(f"{name:<20} {ca:<10.2f} "
          f"{r[0.1]['verified']}/{t} ({100*r[0.1]['verified']/t:5.1f}%)      "
          f"{r[0.5]['verified']}/{t} ({100*r[0.5]['verified']/t:5.1f}%)      "
          f"{r[1.0]['verified']}/{t} ({100*r[1.0]['verified']/t:5.1f}%)")

print(f"\n\n{'='*110}")
print(f"{'Total Loss = Clean CE + λ × Robust CE':^110}")
print(f"{'='*110}")
print(f"{'Model':<20} {'λ=0.1 Loss':<16} {'λ=0.5 Loss':<16} {'λ=1.0 Loss':<16}")
print(f"{'='*110}")
for name in all_results:
    r = all_results[name]["results"]
    t = all_results[name]["total"]
    losses = []
    for lam in lambda_values:
        avg_clean = r[lam]["clean_ce_sum"] / t
        avg_robust = r[lam]["robust_ce_sum"] / t
        losses.append(f"{avg_clean + lam * avg_robust:.4f}")
    print(f"{name:<20} {losses[0]:<16} {losses[1]:<16} {losses[2]:<16}")
print(f"{'='*110}")
