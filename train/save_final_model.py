"""
重新运行完整训练并保存最终模型的脚本

使用方法:
python save_final_model.py --save_path ./models/MNIST_simple_final.pt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'auto_LiRPA'))

from train.spsa_training import train_with_spsa
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and Save Final Model')
    parser.add_argument('--save_path', type=str, 
                        default='./models/MNIST_simple_final.pt',
                        help='Path to save the final model')
    parser.add_argument('--ibp_epochs', type=int, default=3, help='IBP training epochs')
    parser.add_argument('--spsa_epochs', type=int, default=1, help='SPSA epochs (use 1 for quick test)')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Perturbation epsilon')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Running Full Training Pipeline")
    print(f"{'='*70}")
    print(f"IBP Epochs: {args.ibp_epochs}")
    print(f"SPSA Epochs: {args.spsa_epochs}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Save Path: {args.save_path}")
    print(f"{'='*70}\n")
    
    # 运行完整训练流程，指定保存路径
    results = train_with_spsa(
        dataset='MNIST',
        model_type='simple',
        ibp_epochs=args.ibp_epochs,
        spsa_epochs=args.spsa_epochs,
        epsilon=args.epsilon,
        save_path=args.save_path,
        log_dir='./logs'
    )
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Phase 1 (IBP) model saved to: {results['ibp_model_path']}")
    print(f"Final model saved to: {results['final_model_path']}")
    print(f"Total training time: {results['spsa_history']['total_time']:.2f}s")
    print(f"{'='*70}\n")
    
    print(f"\nTo verify the model, run:")
    print(f"python verify_model.py --model_path {args.save_path} --epsilon {args.epsilon}")