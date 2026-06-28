import torch
import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'complete_verifier'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'auto_LiRPA'))

from models import get_mnist_model
from experiments import (
    ABCrownVerifier, load_dataset, run_verification_experiment,
    run_ab_crown_vs_standard, plot_results
)


def load_trained_model(model_path: str):
    model = get_mnist_model('simple')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"  Loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")
    model.eval()
    return model


def main():
    os.makedirs('./results', exist_ok=True)

    models = {
        'IBP': './test_models/MNIST_simple_IBP.pt',
        'CROWN': './test_models/MNIST_simple_CROWN.pt',
    }

    eps_values = [0.01, 0.02, 0.03, 0.05, 0.1]
    sample_size = 20
    timeout = 30

    all_results = pd.DataFrame()

    for label, model_path in models.items():
        print(f"\n{'='*60}")
        print(f"Verifying {label}-trained model")
        print(f"{'='*60}")

        model = load_trained_model(model_path)

        df = run_verification_experiment(model, 'MNIST', sample_size, eps_values, timeout)
        df['training_method'] = label
        all_results = pd.concat([all_results, df], ignore_index=True)

        print(f"\nRunning ablation on {label} model (eps=0.03)...")
        ablation = run_ab_crown_vs_standard(model, 'MNIST', sample_size, 0.03, timeout)
        ablation_serializable = {
            k: {kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv
                for kk, vv in v.items()}
            for k, v in ablation.items()
        }
        with open(f'./results/ablation_{label}.json', 'w') as f:
            json.dump(ablation_serializable, f, indent=2)
        print(f"  αβ-CROWN ablation results saved to ./results/ablation_{label}.json")

    all_results.to_csv('./results/comparison_results.csv', index=False)
    print(f"\nResults saved to ./results/comparison_results.csv")

    print(f"\n{'='*60}")
    print(f"Comparison Summary: IBP vs CROWN")
    print(f"{'='*60}")
    for label in ['IBP', 'CROWN']:
        print(f"\n--- {label}-trained model ---")
        subset = all_results[all_results['training_method'] == label]
        for _, row in subset.iterrows():
            print(f"  eps={row['epsilon']:.2f}: Robust Acc={row['robust_accuracy']:.2f}%, "
                  f"Avg Time={row['avg_time']:.4f}s, "
                  f"Safe={int(row['safe_count'])}, Unsafe={int(row['unsafe_count'])}, "
                  f"Unknown={int(row['unknown_count'])}")


if __name__ == '__main__':
    main()
