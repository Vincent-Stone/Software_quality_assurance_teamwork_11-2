import os
import subprocess
import json
import time

config_dir = os.path.join(os.path.dirname(__file__), 'config')
log_dir = os.path.join(os.path.dirname(__file__), 'log')
results_file = os.path.join(os.path.dirname(__file__), 'verification_results.json')

config_files = sorted([f for f in os.listdir(config_dir) if f.endswith('.yaml')])

print(f"Found {len(config_files)} config files")
for cfg in config_files:
    print(f"  - {cfg}")

results = []

for config_file in config_files:
    config_path = os.path.join(config_dir, config_file)
    model_name = config_file.replace('.yaml', '')
    eps = float(model_name.split('_eps')[1]) / 100.0
    
    log_file = os.path.join(log_dir, f"{model_name}.log")
    
    print(f"\n{'='*60}")
    print(f"Running verification for: {model_name}")
    print(f"  Config: {config_path}")
    print(f"  Log: {log_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        with open(log_file, 'w') as log_f:
            result = subprocess.run(
                ['python', 'abcrown.py', '--config', config_path],
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                capture_output=True,
                text=True,
                timeout=300
            )
            log_f.write(f"=== VERIFICATION START: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log_f.write(f"Model: {model_name}\n")
            log_f.write(f"Epsilon: {eps}\n")
            log_f.write(f"Config file: {config_path}\n")
            log_f.write(f"\n=== STDOUT ===\n")
            log_f.write(result.stdout)
            log_f.write(f"\n=== STDERR ===\n")
            log_f.write(result.stderr)
            log_f.write(f"\n=== VERIFICATION END: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        
        elapsed_time = time.time() - start_time
        
        safe_count = result.stdout.count('safe') + result.stdout.count('SAFE')
        unsafe_count = result.stdout.count('unsafe') + result.stdout.count('UNSAFE')
        timeout_count = result.stdout.count('timeout') + result.stdout.count('TIMEOUT')
        
        results.append({
            'model': model_name,
            'epsilon': eps,
            'status': 'completed' if result.returncode == 0 else 'failed',
            'exit_code': result.returncode,
            'elapsed_time': round(elapsed_time, 2),
            'safe': safe_count,
            'unsafe': unsafe_count,
            'timeout': timeout_count,
            'log_file': log_file
        })
        
        print(f"  Completed in {elapsed_time:.2f}s")
        print(f"  Safe: {safe_count}, Unsafe: {unsafe_count}, Timeout: {timeout_count}")
        
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        with open(log_file, 'w') as log_f:
            log_f.write(f"=== VERIFICATION START: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log_f.write(f"Model: {model_name}\n")
            log_f.write(f"Epsilon: {eps}\n")
            log_f.write(f"\n=== TIMEOUT after {elapsed_time:.2f}s ===\n")
        
        results.append({
            'model': model_name,
            'epsilon': eps,
            'status': 'timeout',
            'exit_code': -1,
            'elapsed_time': round(elapsed_time, 2),
            'safe': 0,
            'unsafe': 0,
            'timeout': 5,
            'log_file': log_file
        })
        
        print(f"  Timeout after {elapsed_time:.2f}s")
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        with open(log_file, 'w') as log_f:
            log_f.write(f"=== VERIFICATION START: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log_f.write(f"Model: {model_name}\n")
            log_f.write(f"Epsilon: {eps}\n")
            log_f.write(f"\n=== ERROR: {str(e)} ===\n")
        
        results.append({
            'model': model_name,
            'epsilon': eps,
            'status': 'error',
            'exit_code': -2,
            'elapsed_time': round(elapsed_time, 2),
            'safe': 0,
            'unsafe': 0,
            'timeout': 0,
            'error_message': str(e),
            'log_file': log_file
        })
        
        print(f"  Error: {str(e)}")

with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"All verifications completed!")
print(f"Results saved to: {results_file}")
print(f"{'='*60}")
