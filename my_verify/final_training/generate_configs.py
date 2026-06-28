import os

models = [
    {'name': 'test_model', 'path': 'models/my_models/final_training/test_model.pt', 'description': 'Standard trained MLP'},
    {'name': 'test_model_ibp', 'path': 'models/my_models/final_training/test_model_ibp.pt', 'description': 'IBP robust trained MLP'},
    {'name': 'test_spsa_model_ibp', 'path': 'models/my_models/final_training/test_spsa_model_ibp.pt', 'description': 'SPSA+IBP robust trained MLP'},
]

epsilons = [0.01, 0.03, 0.05]
test_samples = 5

config_dir = os.path.dirname(__file__) + '/config'
os.makedirs(config_dir, exist_ok=True)

for model in models:
    for eps in epsilons:
        config_content = f'''# Configuration file for {model['name']} with epsilon={eps}
# Description: {model['description']}
# Generated: 2026-06-27

model:
  name: final_training_mlp
  path: {model['path']}

data:
  dataset: MNIST
  mean: [0.0]
  std: [1.0]
  start: 0
  end: {test_samples}

specification:
  norm: .inf
  epsilon: {eps}

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
'''
        config_file = os.path.join(config_dir, f'{model["name"]}_eps{int(eps*100)}.yaml')
        with open(config_file, 'w') as f:
            f.write(config_content)
        print(f'Created: {config_file}')

print(f'\nGenerated {len(models) * len(epsilons)} configuration files')