
# entropy_gate.py - G1/G2 Checkpoint logic
# Part of Reactive Evolution v1

from dataclasses import dataclass
try:
    from laser.uncertainty import UncertaintyVector
    from laser.entropy_profiler import EntropyVector
except ImportError:
    # Fallback/Mock for localized testing without full laser suite
    @dataclass
    class UncertaintyVector:
        h_alea: float = 0.0
        h_epis: float = 0.0
    
    @dataclass
    class EntropyVector:
        h_info: float = 0.0
        h_miss: float = 0.0

class EntropyGate:
    """
    Implements the Checkpoints for the Cell Cycle.
    G1: Availability (Aleatoric Check).
    G2: Safety (Risk Check).
    """

    ALEATORIC_THRESHOLD = 0.4
    RISK_THRESHOLD = 0.3 # Default Risk Tolerance

    def check_g1_availability(self, uncertainty: UncertaintyVector) -> bool:
        """
        G1 Checkpoint: Is the environment stable enough to act?
        """
        if uncertainty.h_alea > self.ALEATORIC_THRESHOLD:
            # Stormy Weather - Wait
            return False
        return True

    def check_g2_safety(self, 
                        entropy: EntropyVector, 
                        uncertainty: UncertaintyVector, 
                        inertia_rank: float) -> bool:
        """
        G2 Checkpoint: Is the proposed change safe?
        
        Risk Model: R = h_epis * (1 - inertia)
        (Actually, High Inertia means HIGH Risk if we break it).
        Correction: The Risk is Impact * Probability.
        Impact = Inertia (High Inertia = High Impact).
        Probability = h_epis (Ignorance).
        
        R = h_epis * inertia_rank
        """
        risk = uncertainty.h_epis * inertia_rank
        
        if risk > self.RISK_THRESHOLD:
            return False # Too risky (Ignorant about Critical System)
            
        return True
