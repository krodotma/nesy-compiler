#!/usr/bin/env python3
# validate_distillation.py - Manual validation script for distillation system

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

from nucleus.tools.distillation.neural_adapter import NeuralAdapter

def validate_examples():
    """Validate the example files to demonstrate the system works."""
    adapter = NeuralAdapter()
    examples_dir = Path(__file__).parent / "examples"
    
    print("=" * 60)
    print("Distillation System Validation")
    print("=" * 60)
    
    examples = [
        ("good_code_example.py", "SHOULD PASS"),
        ("bad_code_example.py", "SHOULD REJECT (anti-patterns)"),
        ("complex_code_example.py", "SHOULD REJECT (complexity)")
    ]
    
    for filename, expected in examples:
        filepath = examples_dir / filename
        if not filepath.exists():
            print(f"\n❌ {filename}: FILE NOT FOUND")
            continue
        
        with open(filepath, 'r') as f:
            code = f.read()
        
        features = adapter.extract_features_from_code(code, str(filepath))
        thrash_prob = adapter.predict_thrash(features)
        
        print(f"\n📄 {filename} ({expected})")
        print(f"   Complexity: {features[0]:.2f}")
        print(f"   AST Depth:  {features[1]:.2f}")
        print(f"   Code Churn: {features[3]:.2f}")
        print(f"   Anti-Pattern: {features[4]:.2f}")
        print(f"   Thrash Prob: {thrash_prob:.2f}")
        
        if "PASS" in expected and thrash_prob < 0.8:
            print(f"   ✅ CORRECT: Would be ACCEPTED")
        elif "REJECT" in expected and thrash_prob >= 0.8:
            print(f"   ✅ CORRECT: Would be REJECTED")
        else:
            print(f"   ❌ WRONG: Unexpected classification")
    
    print("\n" + "=" * 60)
    print("Validation Complete")
    print("=" * 60)

if __name__ == "__main__":
    validate_examples()
