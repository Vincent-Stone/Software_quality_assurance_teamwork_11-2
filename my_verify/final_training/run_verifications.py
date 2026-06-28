import subprocess
import os
import time

models = [
    {'name': 'test_model', 'description': 'Standard trained MLP'},
    {'name': 'test_model_ibp', 'description': 'IBP robust trained MLP'},
    {'name': 'test_spsa_model_ibp', 'description': 'SPSA+IBP robust trained MLP'},
]

epsilons = [0.01, 0.03, 0.05]
eps_labels = ['eps1', 'eps3', 'eps5']

config_dir = os.path.dirname(__file__) + '/config'
log_dir = os.path.dirname(__file__) + '/log'
os.makedirs(log_dir, exist_ok=True)

results = []

for model in models:
    for eps, eps_label in zip(epsilons, eps_labels):
        config_file = f'{config_dir}/{model["name"]}_{eps_label}.yaml'
        log_file = f'{log_dir}/{model["name"]}_{eps_label}.log'
        
        print(f'\n{"="*70}')
        print(f'MODEL: {model["name"]} ({model["description"]})')
        print(f'EPSILON: {eps}')
        print(f'CONFIG: {config_file}')
        print(f'LOG: {log_file}')
        print(f'{"="*70}')
        
        start_time = time.time()
        cmd = ['python', 'abcrown.py', '--config', config_file]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        elapsed_time = time.time() - start_time
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('='*70 + '\n')
            f.write(f'VERIFICATION LOG\n')
            f.write('='*70 + '\n')
            f.write(f'Timestamp: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'Model: {model["name"]}\n')
            f.write(f'Description: {model["description"]}\n')
            f.write(f'Epsilon: {eps}\n')
            f.write(f'Config File: {config_file}\n')
            f.write(f'Elapsed Time: {elapsed_time:.2f} seconds\n')
            f.write(f'Exit Code: {result.returncode}\n')
            f.write('='*70 + '\n\n')
            
            f.write('STDOUT:\n')
            f.write('-'*40 + '\n')
            f.write(result.stdout)
            f.write('\n')
            
            f.write('STDERR:\n')
            f.write('-'*40 + '\n')
            f.write(result.stderr)
        
        stdout_lines = result.stdout.strip().split('\n')
        summary_line = [line for line in stdout_lines if 'Final verified acc' in line]
        time_line = [line for line in stdout_lines if 'mean time for ALL instances' in line]
        status_line = [line for line in stdout_lines if 'safe-incomplete' in line or 'unsafe' in line or 'verified' in line]
        
        acc = summary_line[0] if summary_line else 'N/A'
        avg_time = time_line[0] if time_line else 'N/A'
        status = status_line[-1] if status_line else 'N/A'
        
        results.append({
            'model': model['name'],
            'description': model['description'],
            'epsilon': eps,
            'return_code': result.returncode,
            'accuracy': acc,
            'avg_time': avg_time,
            'status': status,
            'elapsed_time': elapsed_time,
            'log_path': log_file
        })
        
        print(f'Exit Code: {result.returncode}')
        print(f'{acc}')
        print(f'{avg_time}')
        print(f'Status: {status}')
        print(f'Total Time: {elapsed_time:.2f} seconds')

print(f'\n{"="*70}')
print('ALL VERIFICATIONS COMPLETED')
print(f'{"="*70}')

for r in results:
    status_icon = '✓' if r['return_code'] == 0 else '✗'
    print(f"\n{status_icon} {r['model']} (ε={r['epsilon']}):")
    print(f"  Description: {r['description']}")
    print(f"  Status: {'SUCCESS' if r['return_code'] == 0 else 'FAILED'}")
    print(f"  {r['accuracy']}")
    print(f"  {r['avg_time']}")
    print(f"  Overall Status: {r['status']}")
    print(f"  Total Time: {r['elapsed_time']:.2f} seconds")

results_file = os.path.dirname(__file__) + '/verification_results.json'
import json
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to: {results_file}')