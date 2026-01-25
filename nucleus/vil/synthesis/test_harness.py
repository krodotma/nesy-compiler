"""
VIL Synthesis Testing Harness
Comprehensive testing for synthesis components.

Features:
1. CGP evolution testing
2. EGGP evolution testing
3. Program synthesis testing
4. VLM distillation testing
5. Pipeline integration testing
6. CMP integration testing

Version: 1.0
Date: 2026-01-25
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from nucleus.vil.synthesis.evolution import (
    GenotypeType,
    Genotype,
    Phenotype,
    SynthesisResult,
    CartesianGeneticProgramming,
    EGGPEvolver,
    ProgramSynthesis,
    create_program_synthesis,
)
from nucleus.vil.synthesis.pipeline import (
    SynthesisMethod,
    SynthesisRequest,
    SynthesisTracking,
    SynthesisPipelineIntegrator,
    create_synthesis_pipeline_integrator,
)
from nucleus.vil.synthesis.distillation import (
    TeacherModel,
    TeacherOutput,
    DistillationBatch,
    DistillationResult,
    VLMDistiller,
    create_vlm_distiller,
)
from nucleus.vil.cmp.manager import (
    VILCMPManager,
    VisionCMPMetrics,
    create_vil_cmp_manager,
    PHI,
)


class SynthesisTestHarness:
    """
    Testing harness for synthesis components.

    Tests:
    1. CGP: Genome encoding/decoding, mutation, neutral drift
    2. EGGP: Graph creation, mutation, evolution
    3. ProgramSynthesis: Vision-to-code synthesis
    4. PipelineIntegration: End-to-end synthesis with CMP
    5. Distillation: Teacher-student knowledge transfer
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: Dict[str, Any] = {}
        self.passed = 0
        self.failed = 0

    def log(self, message: str) -> None:
        """Log message if verbose."""
        if self.verbose:
            print(f"[SynthesisTestHarness] {message}")

    # ==================== CGP Tests ====================

    def test_cgp_genome_creation(self) -> bool:
        """Test CGP random genome creation."""
        try:
            cgp = CartesianGeneticProgramming(
                num_inputs=5,
                num_outputs=1,
                rows=3,
                cols=5,
                num_functions=5,
            )

            genome = cgp.create_random_genome()

            # Check genome size
            expected_size = 3 * 5 * 2 + 3 * 5  # connections + functions
            assert len(genome) == expected_size, f"Expected size {expected_size}, got {len(genome)}"

            self.log(f"✓ CGP genome creation: size={len(genome)}")
            return True

        except Exception as e:
            self.log(f"✗ CGP genome creation failed: {e}")
            return False

    def test_cgp_genome_decoding(self) -> bool:
        """Test CGP genome decoding to code."""
        try:
            cgp = CartesianGeneticProgramming(
                num_inputs=3,
                num_outputs=1,
                rows=2,
                cols=3,
                num_functions=3,
            )

            genome = cgp.create_random_genome()
            code = cgp.decode_genome(genome)

            # Check code structure
            assert "node_" in code, "Code missing node variables"
            assert "return" in code, "Code missing return statement"

            self.log(f"✓ CGP genome decoding: code_lines={len(code.split(chr(10)))}")
            return True

        except Exception as e:
            self.log(f"✗ CGP genome decoding failed: {e}")
            return False

    def test_cgp_mutation(self) -> bool:
        """Test CGP genome mutation."""
        try:
            cgp = CartesianGeneticProgramming(rows=2, cols=3)
            genome = cgp.create_random_genome()

            # Mutate
            mutated = cgp.mutate_genome(genome, mutation_rate=0.5)

            # Check some genes changed
            changes = np.sum(genome != mutated)
            assert changes > 0, "No mutations occurred"

            self.log(f"✓ CGP mutation: changes={changes}/{len(genome)}")
            return True

        except Exception as e:
            self.log(f"✗ CGP mutation failed: {e}")
            return False

    def test_cgp_neutral_drift(self) -> bool:
        """Test CGP neutral drift."""
        try:
            cgp = CartesianGeneticProgramming(rows=2, cols=3)
            genome = cgp.create_random_genome()

            # Apply neutral drift
            drifted = cgp.neutral_drift(genome, num_steps=5)

            # Check genome changed
            assert len(drifted) == len(genome), "Drift changed genome size"

            self.log(f"✓ CGP neutral drift: steps=5")
            return True

        except Exception as e:
            self.log(f"✗ CGP neutral drift failed: {e}")
            return False

    # ==================== EGGP Tests ====================

    def test_eggp_graph_creation(self) -> bool:
        """Test EGGP random graph creation."""
        try:
            eggp = EGGPEvolver(num_nodes=10)
            adj, node_types = eggp.create_random_graph()

            # Check shapes
            assert adj.shape == (10, 10), f"Adjacency shape wrong: {adj.shape}"
            assert len(node_types) == 10, f"Node types length wrong: {len(node_types)}"

            self.log(f"✓ EGGP graph creation: nodes=10, edges={np.sum(adj)}")
            return True

        except Exception as e:
            self.log(f"✗ EGGP graph creation failed: {e}")
            return False

    def test_eggp_mutation(self) -> bool:
        """Test EGGP graph mutation."""
        try:
            eggp = EGGPEvolver(num_nodes=10)
            adj, node_types = eggp.create_random_graph()

            # Mutate
            adj_mut, types_mut = eggp.mutate_graph(adj, node_types)

            # Check changes
            edge_changes = np.sum(adj != adj_mut)
            type_changes = np.sum(node_types != types_mut)
            total_changes = edge_changes + type_changes

            self.log(f"✓ EGGP mutation: edge_changes={edge_changes}, type_changes={type_changes}")
            return True

        except Exception as e:
            self.log(f"✗ EGGP mutation failed: {e}")
            return False

    def test_eggp_evolution(self) -> bool:
        """Test EGGP evolution."""
        try:
            eggp = EGGPEvolver(num_nodes=10)

            # Create population
            population = [eggp.create_random_graph() for _ in range(5)]
            fitness_scores = [0.5, 0.7, 0.9, 0.6, 0.8]

            # Evolve
            best_adj, best_types = eggp.evolve_generation(population, fitness_scores)

            # Check result
            assert best_adj is not None, "Evolution returned None"
            assert best_types is not None, "Evolution returned None"

            self.log(f"✓ EGGP evolution: population=5")
            return True

        except Exception as e:
            self.log(f"✗ EGGP evolution failed: {e}")
            return False

    # ==================== Program Synthesis Tests ====================

    def test_program_synthesis_cgp(self) -> bool:
        """Test program synthesis with CGP."""
        try:
            synthesizer = create_program_synthesis(method="cgp")

            result = synthesizer.synthesize_from_vision(
                image_data="fake_image_data",
                goal="add two numbers",
                prompt="Write a function to add two numbers",
                num_generations=3,
            )

            # Check result
            assert result.genotype is not None, "Genotype is None"
            assert result.phenotype is not None, "Phenotype is None"
            assert result.phenotype.code is not None, "Code is None"

            self.log(f"✓ Program synthesis CGP: confidence={result.confidence:.3f}")
            return True

        except Exception as e:
            self.log(f"✗ Program synthesis CGP failed: {e}")
            return False

    def test_program_synthesis_eggp(self) -> bool:
        """Test program synthesis with EGGP."""
        try:
            synthesizer = create_program_synthesis(method="eggp")

            result = synthesizer.synthesize_from_vision(
                image_data="fake_image_data",
                goal="click button",
                prompt="Generate action to click button",
                num_generations=3,
            )

            # Check result
            assert result.genotype is not None, "Genotype is None"
            assert result.phenotype is not None, "Phenotype is None"

            self.log(f"✓ Program synthesis EGGP: confidence={result.confidence:.3f}")
            return True

        except Exception as e:
            self.log(f"✗ Program synthesis EGGP failed: {e}")
            return False

    def test_vlm_distillation(self) -> bool:
        """Test VLM distillation."""
        try:
            distiller = create_vlm_distiller()

            # Create teacher outputs
            teacher_outputs = [
                TeacherOutput(
                    teacher=TeacherModel.CLAUDE_OPUS,
                    image_data="img1",
                    prompt="prompt1",
                    response="response1",
                    confidence=0.9,
                    latency_ms=100,
                    timestamp=time.time(),
                ),
                TeacherOutput(
                    teacher=TeacherModel.GEMINI_ULTRA,
                    image_data="img2",
                    prompt="prompt2",
                    response="response2",
                    confidence=0.85,
                    latency_ms=120,
                    timestamp=time.time(),
                ),
            ]

            # Create batch
            batch = DistillationBatch(
                batch_id="test_batch",
                teacher_outputs=teacher_outputs,
            )

            # Distill
            result = asyncio.run(distiller.distill_from_batch(batch))

            # Check result
            assert result.batch_id == "test_batch", "Batch ID mismatch"
            assert len(result.distilled_genotypes) > 0, "No distilled genotypes"

            self.log(f"✓ VLM distillation: num_distilled={len(result.distilled_genotypes)}, alignment={result.teacher_alignment:.3f}")
            return True

        except Exception as e:
            self.log(f"✗ VLM distillation failed: {e}")
            return False

    # ==================== CMP Integration Tests ====================

    def test_cmp_integration(self) -> bool:
        """Test CMP integration with synthesis."""
        try:
            cmp_manager = create_vil_cmp_manager()

            # Create clade with synthesis metrics
            metrics = VisionCMPMetrics(
                capture_quality=0.8,
                analysis_confidence=0.9,
                task_completion=1.0,
                test_coverage=0.9,
            )

            clade_id, clade = cmp_manager.create_clade(metrics=metrics)

            # Check fitness
            fitness = clade.calculate_fitness()
            assert fitness > 0, f"Fitness too low: {fitness}"

            # Check state
            state = clade.update_state()
            assert state is not None, "State is None"

            self.log(f"✓ CMP integration: clade_id={clade_id}, fitness={fitness:.3f}, state={state.value}")
            return True

        except Exception as e:
            self.log(f"✗ CMP integration failed: {e}")
            return False

    def test_phi_weighted_fitness(self) -> bool:
        """Test phi-weighted fitness calculation."""
        try:
            metrics = VisionCMPMetrics(
                capture_quality=0.8,
                analysis_confidence=0.9,
                task_completion=1.0,
                test_coverage=0.9,
            )

            fitness_weighted = metrics.calculate_fitness(phi_weighted=True)
            fitness_unweighted = metrics.calculate_fitness(phi_weighted=False)

            # Weighted should be higher due to PHI multiplier
            assert fitness_weighted > fitness_unweighted, "Weighted fitness should be higher"

            self.log(f"✓ Phi-weighted fitness: weighted={fitness_weighted:.3f}, unweighted={fitness_unweighted:.3f}")
            return True

        except Exception as e:
            self.log(f"✗ Phi-weighted fitness failed: {e}")
            return False

    # ==================== Pipeline Integration Tests ====================

    def test_pipeline_integration(self) -> bool:
        """Test synthesis pipeline integration."""
        try:
            integrator = create_synthesis_pipeline_integrator()

            # Check stats
            stats = integrator.get_stats()
            assert stats is not None, "Stats is None"
            assert "requests_received" in stats, "Missing requests_received"

            self.log(f"✓ Pipeline integration: stats={stats}")
            return True

        except Exception as e:
            self.log(f"✗ Pipeline integration failed: {e}")
            return False

    # ==================== Run All Tests ====================

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all synthesis tests."""
        self.log("=" * 60)
        self.log("VIL Synthesis Test Harness")
        self.log("=" * 60)

        tests = [
            # CGP Tests
            ("CGP Genome Creation", self.test_cgp_genome_creation),
            ("CGP Genome Decoding", self.test_cgp_genome_decoding),
            ("CGP Mutation", self.test_cgp_mutation),
            ("CGP Neutral Drift", self.test_cgp_neutral_drift),
            # EGGP Tests
            ("EGGP Graph Creation", self.test_eggp_graph_creation),
            ("EGGP Mutation", self.test_eggp_mutation),
            ("EGGP Evolution", self.test_eggp_evolution),
            # Program Synthesis Tests
            ("Program Synthesis CGP", self.test_program_synthesis_cgp),
            ("Program Synthesis EGGP", self.test_program_synthesis_eggp),
            ("VLM Distillation", self.test_vlm_distillation),
            # CMP Tests
            ("CMP Integration", self.test_cmp_integration),
            ("Phi-Weighted Fitness", self.test_phi_weighted_fitness),
            # Pipeline Tests
            ("Pipeline Integration", self.test_pipeline_integration),
        ]

        start_time = time.time()

        for name, test_fn in tests:
            try:
                if test_fn():
                    self.passed += 1
                else:
                    self.failed += 1
            except Exception as e:
                self.log(f"✗ {name}: {e}")
                self.failed += 1

        duration = time.time() - start_time

        self.log("=" * 60)
        self.log(f"Results: {self.passed} passed, {self.failed} failed, {duration:.2f}s")
        self.log("=" * 60)

        return {
            "passed": self.passed,
            "failed": self.failed,
            "duration": duration,
            "total": self.passed + self.failed,
        }


def run_synthesis_tests(verbose: bool = True) -> Dict[str, Any]:
    """Run synthesis test harness."""
    harness = SynthesisTestHarness(verbose=verbose)
    return harness.run_all_tests()


__all__ = [
    "SynthesisTestHarness",
    "run_synthesis_tests",
]
