import subprocess
import os

models = [
    ('mnist_cnn7_ibp_eps01', 'mnist_cnn7_ibp_eps01_verify.yaml'),
    ('mnist_cnn_crown', 'mnist_cnn_crown_verify.yaml'),
    ('mnist_cnn_crown_ibp', 'mnist_cnn_crown_ibp_verify.yaml'),
    ('mnist_cnn_fast', 'mnist_cnn_fast_verify.yaml'),
    ('mnist_cnn_ibp', 'mnist_cnn_ibp_verify.yaml'),
]

results = []
log_dir = os.path.join(os.path.dirname(__file__), 'log')
os.makedirs(log_dir, exist_ok=True)

for model_name, config_file in models:
    log_path = os.path.join(log_dir, f'{model_name}.log')
    cmd = ['python', 'abcrown.py', '--config', f'my_verify/{config_file}']
    
    print(f'\n{"="*60}')
    print(f'Running verification for {model_name}...')
    print(f'{"="*60}')
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('STDOUT:\n')
        f.write(result.stdout)
        f.write('\nSTDERR:\n')
        f.write(result.stderr)
    
    print(f'Exit code: {result.returncode}')
    
    stdout_lines = result.stdout.strip().split('\n')
    summary_line = [line for line in stdout_lines if 'Final verified acc' in line]
    time_line = [line for line in stdout_lines if 'mean time for ALL instances' in line]
    
    acc = summary_line[0] if summary_line else 'N/A'
    avg_time = time_line[0] if time_line else 'N/A'
    
    results.append({
        'model': model_name,
        'config': config_file,
        'return_code': result.returncode,
        'accuracy': acc,
        'avg_time': avg_time,
        'log_path': log_path
    })
    
    print(f'{acc}')
    print(f'{avg_time}')

print(f'\n{"="*60}')
print('ALL VERIFICATIONS COMPLETED')
print(f'{"="*60}')
for r in results:
    print(f"\n{r['model']}:")
    print(f"  Status: {'SUCCESS' if r['return_code'] == 0 else 'FAILED'}")
    print(f"  {r['accuracy']}")
    print(f"  {r['avg_time']}")