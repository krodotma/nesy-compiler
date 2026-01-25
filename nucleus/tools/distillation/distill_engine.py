
# distill_engine.py - The Negentropic Distillation Walker
# Production CLI for Pluribus

import sys
import os
import logging
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parents[3]))

from nucleus.tools.reactive_mutator.core import ReactiveMutator
from nucleus.tools.reactive_mutator.entropy_gate import UncertaintyVector, EntropyVector
from nucleus.tools.distillation.triplet_dna import InertiaGate, EntelecheiaGate, HomeostasisGate, DNAContext
from nucleus.tools.distillation.neural_adapter import NeuralAdapter
from nucleus.tools.distillation.kill_switch import KillSwitch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Distiller")

class DistillationEngine:
    def __init__(self, target_root: str, neural_model: str = None):
        self.target_root = Path(target_root)
        self.mutator = ReactiveMutator(str(self.target_root))
        self.neural = NeuralAdapter(neural_model)
        self.kill_switch = KillSwitch(target_root)
        
        # DNA Gates
        self.dna = [
            InertiaGate(),
            EntelecheiaGate(),
            HomeostasisGate()
        ]

    def process_repo(self, source_root: str):
        """
        Walks the Entropic Source and distills valid code to Target.
        """
        source_path = Path(source_root)
        if not source_path.exists():
            logger.error(f"Source not found: {source_root}")
            return

        logger.info(f"Starting Distillation: {source_root} -> {self.target_root}")
        
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.endswith(".py"):
                    self._distill_file(Path(root) / file)

    def _distill_file(self, source_file: Path):
        """
        Applies the Reactive Cycle + Triplet DNA to a single file.
        If accepted, copies to target directory.
        """
        self.kill_switch.check()  # Safety First
        
        rel_path = source_file.relative_to(source_file.parent.parent) if source_file.parent.parent.exists() else source_file.name
        logger.info(f"Processing candidate: {rel_path}")
        
        try:
            # Read source file
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read {rel_path}: {e}")
            return
        
        # 1. Triplet DNA Check
        context = DNAContext(
            source_node=str(source_file), 
            target_graph=None, 
            system_entropy={'h_total': 0.1, 'isolation_score': 0.0}
        )
        
        for gate in self.dna:
            if not gate.check(context):
                logger.warning(f"❌ REJECTED by {gate.name()}: {rel_path}")
                return

        # 2. Neural Gate Check - Use enhanced feature extraction
        feature_vector = self.neural.extract_features_from_code(content, str(source_file))
        thrash_prob = self.neural.predict_thrash(feature_vector)
        
        logger.info(f"Neural Analysis: complexity={feature_vector[0]:.2f}, " +
                   f"depth={feature_vector[1]:.2f}, anti_pattern={feature_vector[4]:.2f}, " +
                   f"thrash_prob={thrash_prob:.2f}")
        
        if thrash_prob > 0.8:
            logger.warning(f"❌ REJECTED by NeuralGate (Prob {thrash_prob:.2f}): {rel_path}")
            return

        # 3. Copy to Target (Distillation)
        target_file = self.target_root / rel_path
        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(content, encoding='utf-8')
            logger.info(f"✅ ACCEPTED & COPIED: {rel_path} -> {target_file}")
        except Exception as e:
            logger.error(f"Failed to copy {rel_path} to target: {e}")
            return 

def main():
    parser = argparse.ArgumentParser(description="Pluribus Distillation Engine")
    parser.add_argument("--source", required=True, help="Entropic Source Directory")
    parser.add_argument("--target", required=True, help="Negentropic Target Directory")
    parser.add_argument("--neural", help="Path to Neural Gate Model")
    
    args = parser.parse_args()
    
    engine = DistillationEngine(args.target, args.neural)
    engine.process_repo(args.source)

if __name__ == "__main__":
    main()
