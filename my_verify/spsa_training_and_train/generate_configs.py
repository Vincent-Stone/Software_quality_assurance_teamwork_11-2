import os

models = [
    {
        'name': 'MNIST_simple_final_ibp',
        'file': 'models/my_models/spsa_training/MNIST_simple_final_ibp.pt',
        'model_def': 'spsa_training_mlp'
    },
    {
        'name': 'MNIST_simple_CROWN-IBP_wcim',
        'file': 'models/my_models/spsa_training/MNIST_simple_CROWN-IBP_wcim.pt',
        'model_def': 'spsa_training_mlp'
    }
]

epsilons = [0.01, 0.03, 0.05]

config_template = """# Configuration file for {model_name} with epsilon={epsilon}
# Description: SPSA training model verification
# Generated: 2026-06-27

model:
  name: {model_def}
  path: {model_file}

data:
  dataset: MNIST
  mean: [0.0]
  std: [1.0]
  start: 0
  end: 5

specification:
  norm: .inf
  epsilon: {epsilon}

attack:
  pgd_steps: 100
  pgd_restarts: 10

solver:
  batch_size: 1024
  alpha-crown:
    iteration: 100
    lr_alpha: 0.1
  beta-crown:
    iteration: 20
    lr_alpha: 0.01
    lr_beta: 0.05

bab:
  timeout: 120
  branching:
    reduceop: min
    method: kfsb
    candidates: 3

general:
  device: cpu
"""

config_dir = os.path.join(os.path.dirname(__file__), 'config')
log_dir = os.path.join(os.path.dirname(__file__), 'log')

os.makedirs(config_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

for model in models:
    for eps in epsilons:
        eps_suffix = f'eps{int(eps * 100)}'
        config_filename = f"{model['name']}_{eps_suffix}.yaml"
        
        config_content = config_template.format(
            model_name=model['name'],
            epsilon=eps,
            model_def=model['model_def'],
            model_file=model['file']
        )
        
        config_path = os.path.join(config_dir, config_filename)
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"Created config: {config_path}")

print(f"\nGenerated {len(models) * len(epsilons)} config files")
