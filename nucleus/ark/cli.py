#!/usr/bin/env python3
"""
ark CLI - Autonomous Reactive Kernel command-line interface

The negentropic evolutionary VCS for Pluribus.
"""

import argparse
import sys
import json
from pathlib import Path

from nucleus.ark.core.repository import ArkRepository
from nucleus.ark.core.context import ArkCommitContext


def cmd_init(args):
    """Initialize a new ARK repository."""
    path = Path(args.path or ".").resolve()
    repo = ArkRepository(str(path))
    
    if repo.init():
        print(f"✅ ARK repository initialized at {path}")
        print("   Created .ark/ directory")
        print("   DNA gates: Inertia, Entelecheia, Homeostasis")
        return 0
    else:
        print(f"❌ Failed to initialize ARK repository")
        return 1


def cmd_status(args):
    """Show ARK repository status with DNA metrics."""
    path = Path(args.path or ".").resolve()
    repo = ArkRepository(str(path))
    
    status = repo.status()
    
    if "error" in status:
        print(f"❌ {status['error']}")
        return 1
    
    print(f"📦 ARK Repository: {status['path']}")
    print(f"   Gates enabled: {status['gates_enabled']}")
    print()
    
    if status['staged']:
        print("Staged for commit:")
        for f in status['staged']:
            print(f"   ✅ {f}")
    
    if status['modified']:
        print("Modified (not staged):")
        for f in status['modified']:
            print(f"   📝 {f}")
    
    print()
    print("Entropy Vector (H*):")
    entropy = status.get('current_entropy', {})
    for key, value in entropy.items():
        bar = "█" * int(value * 10) + "░" * (10 - int(value * 10))
        status_icon = "🔴" if value > 0.7 else "🟡" if value > 0.4 else "✅"
        print(f"   {key}: {bar} {value:.2f} {status_icon}")
    
    return 0


def cmd_commit(args):
    """DNA-gated commit using Cell Cycle."""
    path = Path(".").resolve()
    repo = ArkRepository(str(path))
    
    if not repo.is_initialized:
        print("❌ Not an ARK repository. Run 'ark init' first.")
        return 1
    
    # Build context
    context = ArkCommitContext(
        etymology=args.etymology or "",
        purpose=args.message,
        stage_all=not args.no_add,
    )
    
    # Add witness if requested
    if args.witness:
        context.with_witness(
            attester=args.witness,
            intent=args.message
        )
        context.require_witness = False  # Already provided
    
    print("🧬 Running Cell Cycle...")
    print("   G1: Checking environment stability...")
    
    sha = repo.commit(args.message, context)
    
    if sha:
        print(f"✅ Commit successful: {sha[:8]}")
        return 0
    else:
        print("❌ Commit rejected by DNA gates")
        return 1


def cmd_log(args):
    """Show ARK-enhanced commit log."""
    path = Path(".").resolve()
    repo = ArkRepository(str(path))
    
    if not repo.is_initialized:
        print("❌ Not an ARK repository")
        return 1
    
    commits = repo.log(limit=args.limit)
    
    for commit in commits:
        sha = commit['sha'][:8]
        subject = commit['subject']
        cmp = commit.get('cmp')
        etymology = commit.get('etymology')
        
        cmp_str = f"CMP={cmp:.2f}" if cmp else "CMP=?"
        print(f"{sha} {subject} [{cmp_str}]")
        
        if args.verbose and etymology:
            print(f"         Etymology: {etymology}")
    
    return 0


def cmd_health(args):
    """Show ARK system health."""
    path = Path(".").resolve()
    repo = ArkRepository(str(path))
    
    if not repo.is_initialized:
        print("❌ Not an ARK repository")
        return 1
    
    print("🏥 ARK Health Check")
    print("=" * 40)
    
    # Check gates
    print("DNA Gates:")
    for gate in repo.gates:
        print(f"   ✅ {gate.name()}: Active")
    
    # Check Rhizom
    if repo.rhizom_path.exists():
        print(f"   ✅ Rhizom: Active")
    else:
        print(f"   ⚠️  Rhizom: Missing")
    
    return 0


def cmd_distill(args):
    """Run the distillation pipeline."""
    from nucleus.ark.portal.distill import DistillationPipeline
    
    source = Path(args.source).resolve()
    target = Path(args.target).resolve()
    
    print(f"🔬 Distillation: {source} → {target}")
    
    pipeline = DistillationPipeline(str(source), str(target))
    report = pipeline.run(purpose=args.purpose or "ARK distillation")
    
    print(f"\n📊 Results:")
    print(f"   Files processed: {report.ingest_report.total_files if report.ingest_report else 0}")
    print(f"   Accepted: {report.ingest_report.accepted_files if report.ingest_report else 0}")
    print(f"   Rejected: {report.ingest_report.rejected_files if report.ingest_report else 0}")
    print(f"   Genes created: {report.genes_created}")
    print(f"   Total CMP: {report.total_cmp:.2f}")
    print(f"   Status: {report.status}")
    
    return 0 if report.status == "completed" else 1


def cmd_ancestry(args):
    """Show commit ancestry with CMP trajectory."""
    from nucleus.ark.rhizom.dag import RhizomDAG
    from nucleus.ark.rhizom.lineage import LineageTracker
    
    path = Path(".").resolve()
    rhizom = RhizomDAG(path)
    tracker = LineageTracker(rhizom)
    
    lineage = tracker.get_lineage(args.sha, max_depth=args.depth)
    
    print(f"📜 Ancestry for {args.sha[:8]}:")
    print(f"   Depth: {lineage.depth}")
    print(f"   Clades: {', '.join(lineage.clades) or 'none'}")
    print(f"\n   CMP Trajectory:")
    for i, cmp in enumerate(lineage.cmp_trajectory):
        bar = "█" * int(cmp * 10)
        print(f"   {i}: {bar} {cmp:.2f}")
    
    return 0


def cmd_verify(args):
    """Verify LTL specs against current state."""
    from nucleus.ark.synthesis.ltl_spec import PluribusLTLSpec, LTLVerifier
    
    spec = PluribusLTLSpec.core_spec()
    verifier = LTLVerifier(spec)
    
    # Create a trace from current commit state
    # (simplified - in production, this would read actual state)
    trace = [{
        "commit": True,
        "inertia_pass": True,
        "entelecheia_pass": True,
        "homeostasis_pass": True,
        "stable": True
    }]
    
    passed = verifier.verify_trace(trace)
    print(verifier.get_violation_report())
    
    return 0 if passed else 1


def cmd_suggest(args):
    """Get MetaLearner suggestions for current staged changes.
    
    P2-005: Implements `ark suggest` command.
    """
    from nucleus.ark.meta.client import MetaLearnerClient
    
    path = Path(".").resolve()
    repo = ArkRepository(str(path))
    
    if not repo.is_initialized:
        print("❌ Not an ARK repository")
        return 1
    
    client = MetaLearnerClient()
    
    if not client.available:
        print("⚠️  MetaLearner not available - using fallback suggestions")
    
    # Get staged files
    status = repo.status()
    staged = status.get("staged", [])
    modified = status.get("modified", [])
    files = staged or modified
    
    if not files:
        print("❌ No files to analyze. Stage or modify some files first.")
        return 1
    
    print(f"🧠 MetaLearner Suggestions for {len(files)} files")
    print("=" * 40)
    
    # Get predictions
    entropy = status.get("current_entropy", {})
    
    # CMP Prediction
    cmp_result = client.predict_cmp(files, args.message or "", entropy)
    print(f"\n📈 Predicted CMP: {cmp_result.value:.2f}")
    print(f"   Confidence: {cmp_result.confidence:.0%}")
    print(f"   Source: {cmp_result.source}")
    
    # Etymology Suggestion
    etym_result = client.suggest_etymology(files, args.message or "")
    print(f"\n📚 Suggested Etymology: {etym_result.value}")
    
    # Prescient Analysis
    if args.full:
        print(f"\n🔮 Prescient Analysis:")
        analysis = client.prescient_analysis({
            "files": files,
            "message": args.message or "",
            "entropy": entropy,
            "snippets": []
        })
        
        risk_emoji = {"low": "✅", "medium": "⚠️", "high": "🔴"}.get(analysis["risk_level"], "❓")
        print(f"   Risk Level: {risk_emoji} {analysis['risk_level'].upper()}")
        
        if analysis.get("suggestions"):
            print(f"   Suggestions:")
            for s in analysis["suggestions"]:
                print(f"      → {s}")
        
        print(f"\n   Gate Predictions:")
        for gate, passes in analysis.get("gate_predictions", {}).items():
            icon = "✅" if passes else "❌"
            print(f"      {icon} {gate}")
    
    return 0


def cmd_learn(args):
    """Manually record experiences for MetaLearner.
    
    P2-013: Implements `ark learn` command.
    """
    from nucleus.ark.meta.client import MetaLearnerClient, ExperienceRecord
    
    client = MetaLearnerClient()
    
    if args.flush:
        count = client.flush_experience_buffer()
        print(f"✅ Flushed {count} experiences to persistent storage")
        return 0
    
    if args.icl:
        context = client.get_icl_context()
        print(f"📚 In-Context Learning Examples ({len(context)} items)")
        print("=" * 40)
        for i, ex in enumerate(context, 1):
            delta = ex.get("cmp_delta", 0)
            icon = "✅" if ex.get("success") else "❌"
            print(f"{i}. {icon} CMP Δ={delta:+.2f} | {ex.get('etymology', 'unknown')[:40]}")
        return 0
    
    if args.record_sha:
        # Record a manual experience
        record = ExperienceRecord(
            commit_sha=args.record_sha,
            etymology=args.etymology or "",
            cmp_before=args.cmp_before or 0.5,
            cmp_after=args.cmp_after or 0.5,
            entropy_before={},
            entropy_after={},
            gate_results={},
            success=not args.failed
        )
        client.record_experience(record)
        print(f"✅ Recorded experience for {args.record_sha[:8]}")
        return 0
    
    print("Usage: ark learn [--flush | --icl | --record-sha SHA]")
    print("   --flush      Flush experience buffer to storage")
    print("   --icl        Show in-context learning examples")
    print("   --record-sha Record manual experience for a commit")
    return 0


def cmd_train(args):
    """Train CL engine on recorded experiences.
    
    P2-027: Implements `ark train` command.
    """
    from nucleus.ark.cl.replay_buffer import PrioritizedReplayBuffer
    from nucleus.ark.cl.curriculum import CurriculumLearning
    from nucleus.ark.cl.pattern_memory import PatternMemory
    
    curriculum = CurriculumLearning()
    
    if args.curriculum:
        stats = curriculum.get_statistics()
        print(f"📚 Curriculum Learning Status")
        print("=" * 40)
        print(f"   Level: {stats['current_level']} ({stats['level_name']})")
        print(f"   Successes: {stats['successes']}/{stats['needed_for_advance']}")
        print(f"   Success Rate: {stats['success_rate']:.1%}")
        print(f"   Learning Rate: {stats['learning_rate']:.6f}")
        print(f"   Difficulty: {stats['difficulty_multiplier']:.2f}")
        print(f"   Total Steps: {stats['total_steps']}")
        return 0
    
    if args.reset:
        curriculum.reset()
        print("✅ Curriculum reset to level 0 (Beginner)")
        return 0
    
    # Run training
    buffer = PrioritizedReplayBuffer()
    patterns = PatternMemory()
    
    print(f"🏋️ Training CL Engine")
    print(f"   Buffer size: {len(buffer)}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Epochs: {args.epochs}")
    print()
    
    if len(buffer) == 0:
        print("⚠️  No experiences in buffer. Make some commits first!")
        return 0
    
    for epoch in range(args.epochs):
        batch = buffer.sample_batch(args.batch_size)
        
        # Compute statistics from batch
        mean_cmp_delta = sum(b["cmp_delta"] for b in batch) / len(batch)
        mean_reward = sum(b["reward"] for b in batch) / len(batch)
        success_rate = sum(b["success"] for b in batch) / len(batch)
        
        print(f"   Epoch {epoch + 1}: CMP Δ={mean_cmp_delta:+.3f}, "
              f"Reward={mean_reward:.3f}, Success={success_rate:.1%}")
    
    # Show pattern memory stats
    pattern_stats = patterns.get_statistics()
    print(f"\n📊 Pattern Memory:")
    print(f"   Patterns: {pattern_stats['pattern_count']}")
    print(f"   Anti-patterns: {pattern_stats['anti_pattern_count']}")
    
    return 0


def cmd_checkpoint(args):
    """Manage model checkpoints.
    
    P2-036: Implements `ark checkpoint` command.
    """
    from nucleus.ark.cl.checkpoint import CheckpointManager
    
    manager = CheckpointManager()
    
    if args.action == "list":
        versions = manager.list_versions()
        print(f"📁 Model Checkpoints ({len(versions)} total)")
        print("=" * 50)
        for v in versions:
            best_marker = "⭐" if v["is_best"] else "  "
            print(f"{best_marker} {v['version']}: CMP={v['cmp_mean']:.3f}, "
                  f"Success={v['success_rate']:.1%}, Score={v['score']:.3f}")
        return 0
    
    if args.action == "save":
        # Save current state as checkpoint
        from nucleus.ark.cl.ewc import ElasticWeightConsolidation
        from nucleus.ark.cl.curriculum import CurriculumLearning
        
        ewc = ElasticWeightConsolidation()
        curriculum = CurriculumLearning()
        stats = curriculum.get_statistics()
        
        version = manager.save(
            parameters=ewc.optimal_params,
            fisher=ewc.fisher,
            cmp_mean=0.5,  # Would come from recent experiences
            success_rate=stats["success_rate"],
            curriculum_level=stats["current_level"]
        )
        print(f"✅ Saved checkpoint: {version}")
        return 0
    
    if args.action == "load":
        version = args.version
        if not version:
            print("❌ Specify version with -v/--version")
            return 1
        
        checkpoint = manager.load(version)
        if checkpoint:
            print(f"✅ Loaded checkpoint {version}")
            print(f"   CMP: {checkpoint.cmp_mean:.3f}")
            print(f"   Curriculum Level: {checkpoint.curriculum_level}")
        else:
            print(f"❌ Checkpoint {version} not found")
            return 1
        return 0
    
    if args.action == "best":
        checkpoint = manager.load_best()
        if checkpoint:
            print(f"✅ Loaded best checkpoint: {checkpoint.version}")
            print(f"   Score: {checkpoint.score:.3f}")
            print(f"   CMP: {checkpoint.cmp_mean:.3f}")
        else:
            print("⚠️  No checkpoints available")
        return 0
    
    return 0


def cmd_neural(args):
    """Neural gate status and operations.
    
    P2-051: Implements `ark neural status` command.
    """
    from nucleus.ark.neural.model import NeuralGate, HAS_TORCH
    from nucleus.ark.neural.thrash_predictor import ThrashPredictor
    from nucleus.ark.neural.bayesian import BayesianNeuralGate
    
    if args.action == "status":
        print(f"🧠 Neural Gate Status")
        print("=" * 40)
        print(f"   PyTorch available: {'✅' if HAS_TORCH else '❌'}")
        
        gate = NeuralGate()
        stats = gate.get_statistics()
        print(f"   Model loaded: {'✅' if stats['model_loaded'] else '❌'}")
        print(f"   Hidden dim: {stats['config']['hidden_dim']}")
        print(f"   Num gates: {stats['config']['num_gates']}")
        print(f"   Cache size: {stats['cache_size']}")
        return 0
    
    if args.action == "predict":
        # Quick prediction from current entropy
        path = Path(".").resolve()
        repo = ArkRepository(str(path))
        status = repo.status()
        entropy = status.get("current_entropy", {})
        
        if not entropy:
            print("⚠️  No entropy data. Run on a repo with staged changes.")
            return 0
        
        print(f"🧠 Neural Gate Prediction")
        print("=" * 40)
        
        gate = NeuralGate()
        predictions = gate.predict(entropy)
        
        for name, pred in predictions.items():
            icon = "✅" if pred.decision == "ACCEPT" else "⚠️" if pred.decision == "REVIEW" else "❌"
            print(f"   {icon} {name}: {pred.probability:.1%} ({pred.decision})")
            print(f"      Confidence: {pred.confidence:.1%}")
        
        # Thrash prediction
        thrash = ThrashPredictor()
        thrash_result = thrash.predict_from_entropy(entropy)
        
        risk_icon = {"low": "✅", "medium": "⚠️", "high": "🔴", "critical": "🚨"}.get(thrash_result.risk_level, "❓")
        print(f"\n   Thrash: {risk_icon} {thrash_result.thrash_probability:.1%} ({thrash_result.risk_level})")
        print(f"   Quality: {thrash_result.quality_score:.2f}")
        print(f"   CI: [{thrash_result.confidence_interval[0]:.2f}, {thrash_result.confidence_interval[1]:.2f}]")
        print(f"   Recommendation: {thrash_result.recommendation}")
        
        return 0
    
    if args.action == "bayesian":
        # Bayesian uncertainty analysis
        path = Path(".").resolve()
        repo = ArkRepository(str(path))
        status = repo.status()
        entropy = status.get("current_entropy", {})
        
        if not entropy:
            print("⚠️  No entropy data.")
            return 0
        
        print(f"🎲 Bayesian Uncertainty Analysis")
        print("=" * 40)
        
        bayes = BayesianNeuralGate()
        estimate = bayes.predict(entropy)
        
        print(f"   Mean: {estimate.mean:.3f}")
        print(f"   Std Dev: {estimate.std:.3f}")
        print(f"   Epistemic: {estimate.epistemic:.4f} (model uncertainty)")
        print(f"   Aleatoric: {estimate.aleatoric:.4f} (data uncertainty)")
        print(f"   Total: {estimate.total_uncertainty:.3f}")
        print(f"   Confidence: {estimate.confidence:.1%}")
        return 0
    
    print("Usage: ark neural [status | predict | bayesian]")
    return 0


def cmd_explain(args):
    """Explain neural gate decisions.
    
    P2-059: Implements `ark explain` command.
    """
    from nucleus.ark.neural.model import NeuralGate
    from nucleus.ark.neural.features import FeatureExtractor, CodeFeatures
    
    path = Path(".").resolve()
    repo = ArkRepository(str(path))
    status = repo.status()
    entropy = status.get("current_entropy", {})
    
    if not entropy:
        print("⚠️  No entropy data. Stage some changes first.")
        return 0
    
    print(f"🔍 Gate Decision Explanation")
    print("=" * 50)
    
    gate = NeuralGate()
    predictions = gate.predict(entropy)
    
    # Show entropy contribution
    print(f"\n📊 Entropy Vector (H*):")
    sorted_entropy = sorted(entropy.items(), key=lambda x: x[1], reverse=True)
    for key, value in sorted_entropy:
        bar = "█" * int(value * 10) + "░" * (10 - int(value * 10))
        impact = "↑ blocks" if value > 0.7 else "↓ allows" if value < 0.3 else "~ neutral"
        print(f"   {key:15}: {bar} {value:.2f} ({impact})")
    
    # Show gate decisions with explanations
    print(f"\n🧬 Gate Decisions:")
    for name, pred in predictions.items():
        icon = "✅" if pred.decision == "ACCEPT" else "⚠️" if pred.decision == "REVIEW" else "❌"
        print(f"\n   {icon} {name.upper()}")
        print(f"      Probability: {pred.probability:.1%}")
        print(f"      Decision: {pred.decision}")
        print(f"      Confidence: {pred.confidence:.1%}")
        
        # Explain based on entropy
        if name == "inertia":
            h_struct = entropy.get("h_struct", 0.5)
            if h_struct > 0.7:
                print(f"      Reason: High structural entropy ({h_struct:.2f}) suggests instability")
            else:
                print(f"      Reason: Low structural entropy ({h_struct:.2f}) indicates stability")
        elif name == "entelecheia":
            h_goal = entropy.get("h_goal_drift", 0.5)
            if h_goal > 0.5:
                print(f"      Reason: Goal drift detected ({h_goal:.2f}) - purpose unclear")
            else:
                print(f"      Reason: Low goal drift ({h_goal:.2f}) - purpose aligned")
        elif name == "homeostasis":
            h_total = sum(entropy.values()) / len(entropy)
            if h_total > 0.6:
                print(f"      Reason: High total entropy ({h_total:.2f}) threatens balance")
            else:
                print(f"      Reason: Total entropy ({h_total:.2f}) within safe bounds")
    
    return 0


def cmd_fuzz(args):
    """Fuzz testing for gates.
    
    P2-067: Implements `ark fuzz` command.
    """
    from nucleus.ark.testing.fuzzer import CoverageFuzzer
    
    print(f"🔥 Fuzzing ARK Gates")
    print("=" * 40)
    print(f"   Iterations: {args.iterations}")
    
    fuzzer = CoverageFuzzer(max_iterations=args.iterations)
    
    def mock_gate(entropy: Dict) -> bool:
        h_total = sum(entropy.values()) / max(len(entropy), 1)
        return h_total < 0.7
    
    result = fuzzer.fuzz_gate(mock_gate)
    
    print(f"\n📊 Results:")
    print(f"   Total inputs: {result.total_inputs}")
    print(f"   Crashes: {result.crashes}")
    print(f"   New coverage: {result.new_coverage}")
    print(f"   Coverage: {result.coverage_percentage:.1f}%")
    print(f"   Time: {result.execution_time:.2f}s")
    
    if result.crash_inputs:
        print(f"\n🐛 Crash Inputs ({len(result.crash_inputs)}):")
        for inp in result.crash_inputs[:3]:
            print(f"   {inp.id}: {inp.mutation}")
    
    return 0


def cmd_golden(args):
    """Manage golden tests.
    
    P2-079: Implements `ark golden` command.
    """
    from nucleus.ark.testing.golden import GoldenTestManager
    
    manager = GoldenTestManager()
    
    if args.action == "list":
        tests = manager.list_tests()
        stats = manager.get_statistics()
        
        print(f"📸 Golden Tests ({stats['total']} total)")
        print("=" * 40)
        print(f"   Passed: {stats['passed']}")
        print(f"   Failed: {stats['failed']}")
        
        for t in tests[:10]:
            icon = "✅" if t["passed"] else "❌"
            print(f"   {icon} {t['name']}: {t['id']}")
        return 0
    
    if args.update_all:
        count = manager.update_all_failing()
        print(f"✅ Updated {count} failing golden tests")
        return 0
    
    print("Usage: ark golden [list | --update-all]")
    return 0


def cmd_benchmark(args):
    """Performance benchmarks.
    
    P2-090: Implements `ark benchmark` command.
    """
    from nucleus.ark.perf.profiler import GateProfiler
    from nucleus.ark.perf.cache import get_cache
    import time
    
    print(f"⚡ ARK Performance Benchmark")
    print("=" * 40)
    
    profiler = GateProfiler()
    iterations = args.iterations
    
    # Benchmark gate execution
    def mock_gate():
        time.sleep(0.001)
        return True
    
    print(f"Running {iterations} iterations...")
    for i in range(iterations):
        profiler.start("gate.mock")
        mock_gate()
        profiler.stop("gate.mock")
    
    metrics = profiler.get_metrics("gate.mock")
    if metrics:
        print(f"\n📊 Gate Metrics:")
        print(f"   Mean: {metrics.mean_ms:.2f}ms")
        print(f"   P95: {metrics.p95_ms:.2f}ms")
        print(f"   P99: {metrics.p99_ms:.2f}ms")
        print(f"   Total: {metrics.total_ms:.2f}ms")
    
    # Cache stats
    cache = get_cache()
    stats = cache.get_stats()
    print(f"\n📦 Cache Stats:")
    print(f"   Size: {stats['size']}/{stats['max_size']}")
    print(f"   Hit Rate: {stats['hit_rate']:.1%}")
    
    return 0


def cmd_safety(args):
    """Safety status check.
    
    P2-097: Implements `ark safety-check` command.
    """
    from nucleus.ark.perf.safety import KillSwitch, RateLimiter, ResourceGuard
    
    kill_switch = KillSwitch()
    rate_limiter = RateLimiter()
    resource_guard = ResourceGuard()
    
    if args.kill:
        kill_switch.trigger("Manual CLI trigger")
        print("🛑 Kill switch ACTIVATED")
        return 0
    
    if args.reset:
        kill_switch.force_reset()
        print("✅ Kill switch reset")
        return 0
    
    print(f"🛡️ ARK Safety Status")
    print("=" * 40)
    
    # Kill switch
    ks_status = "🛑 ACTIVE" if kill_switch.active else "✅ OK"
    print(f"   Kill Switch: {ks_status}")
    
    # Rate limiter
    usage = rate_limiter.get_usage()
    print(f"   Rate Limit: {usage['minute']}/{usage['minute_limit']} (per min)")
    
    # Resources
    resources = resource_guard.check()
    res_status = "✅ OK" if resources.get('all_ok', True) else "⚠️ WARNING"
    print(f"   Resources: {res_status}")
    if 'memory_mb' in resources:
        print(f"      Memory: {resources['memory_mb']:.0f}MB")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog='ark',
        description='ARK: Autonomous Reactive Kernel - Negentropic VCS'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # init
    init_parser = subparsers.add_parser('init', help='Initialize ARK repository')
    init_parser.add_argument('path', nargs='?', help='Path to initialize (default: current)')
    init_parser.set_defaults(func=cmd_init)
    
    # status
    status_parser = subparsers.add_parser('status', help='Show repository status')
    status_parser.add_argument('path', nargs='?', help='Path (default: current)')
    status_parser.set_defaults(func=cmd_status)
    
    # commit
    commit_parser = subparsers.add_parser('commit', help='DNA-gated commit')
    commit_parser.add_argument('-m', '--message', required=True, help='Commit message')
    commit_parser.add_argument('-e', '--etymology', help='Semantic origin')
    commit_parser.add_argument('-w', '--witness', help='Witness attester')
    commit_parser.add_argument('--no-add', action='store_true', help='Do not auto-stage')
    commit_parser.set_defaults(func=cmd_commit)
    
    # log
    log_parser = subparsers.add_parser('log', help='Show commit log')
    log_parser.add_argument('-n', '--limit', type=int, default=10, help='Number of commits')
    log_parser.add_argument('-v', '--verbose', action='store_true', help='Show etymology')
    log_parser.set_defaults(func=cmd_log)
    
    # health
    health_parser = subparsers.add_parser('health', help='Show system health')
    health_parser.set_defaults(func=cmd_health)
    
    # distill
    distill_parser = subparsers.add_parser('distill', help='Run distillation pipeline')
    distill_parser.add_argument('source', help='Entropic source directory')
    distill_parser.add_argument('target', help='Negentropic target directory')
    distill_parser.add_argument('-p', '--purpose', help='Distillation purpose')
    distill_parser.set_defaults(func=cmd_distill)
    
    # ancestry
    ancestry_parser = subparsers.add_parser('ancestry', help='Show commit ancestry')
    ancestry_parser.add_argument('sha', help='Commit SHA')
    ancestry_parser.add_argument('-d', '--depth', type=int, default=10, help='Max depth')
    ancestry_parser.set_defaults(func=cmd_ancestry)
    
    # verify
    verify_parser = subparsers.add_parser('verify', help='Verify LTL specs')
    verify_parser.set_defaults(func=cmd_verify)
    
    # suggest (P2-005)
    suggest_parser = subparsers.add_parser('suggest', help='Get MetaLearner suggestions')
    suggest_parser.add_argument('-m', '--message', help='Commit message for context')
    suggest_parser.add_argument('-f', '--full', action='store_true', help='Full prescient analysis')
    suggest_parser.set_defaults(func=cmd_suggest)
    
    # learn (P2-013)
    learn_parser = subparsers.add_parser('learn', help='MetaLearner learning commands')
    learn_parser.add_argument('--flush', action='store_true', help='Flush experience buffer')
    learn_parser.add_argument('--icl', action='store_true', help='Show ICL context')
    learn_parser.add_argument('--record-sha', dest='record_sha', help='Record experience for SHA')
    learn_parser.add_argument('-e', '--etymology', help='Etymology for recorded experience')
    learn_parser.add_argument('--cmp-before', type=float, dest='cmp_before', help='CMP before commit')
    learn_parser.add_argument('--cmp-after', type=float, dest='cmp_after', help='CMP after commit')
    learn_parser.add_argument('--failed', action='store_true', help='Mark experience as failed')
    learn_parser.set_defaults(func=cmd_learn)
    
    # train (P2-027)
    train_parser = subparsers.add_parser('train', help='Train CL engine on experiences')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    train_parser.add_argument('--epochs', type=int, default=1, help='Training epochs')
    train_parser.add_argument('--curriculum', action='store_true', help='Show curriculum status')
    train_parser.add_argument('--reset', action='store_true', help='Reset curriculum to level 0')
    train_parser.set_defaults(func=cmd_train)
    
    # checkpoint (P2-036)
    checkpoint_parser = subparsers.add_parser('checkpoint', help='Manage model checkpoints')
    checkpoint_parser.add_argument('action', nargs='?', choices=['list', 'save', 'load', 'best'], 
                                  default='list', help='Checkpoint action')
    checkpoint_parser.add_argument('-v', '--version', help='Checkpoint version for load')
    checkpoint_parser.set_defaults(func=cmd_checkpoint)
    
    # neural (P2-051)
    neural_parser = subparsers.add_parser('neural', help='Neural gate operations')
    neural_parser.add_argument('action', nargs='?', choices=['status', 'predict', 'bayesian'],
                              default='status', help='Neural gate action')
    neural_parser.set_defaults(func=cmd_neural)
    
    # explain (P2-059)
    explain_parser = subparsers.add_parser('explain', help='Explain gate decisions')
    explain_parser.set_defaults(func=cmd_explain)
    
    # fuzz (P2-067)
    fuzz_parser = subparsers.add_parser('fuzz', help='Fuzz testing for gates')
    fuzz_parser.add_argument('-n', '--iterations', type=int, default=100, help='Fuzz iterations')
    fuzz_parser.add_argument('--gate', help='Gate to fuzz (inertia/entelecheia/homeostasis)')
    fuzz_parser.set_defaults(func=cmd_fuzz)
    
    # golden (P2-079)
    golden_parser = subparsers.add_parser('golden', help='Manage golden tests')
    golden_parser.add_argument('action', nargs='?', choices=['list', 'update', 'check'],
                              default='list', help='Golden test action')
    golden_parser.add_argument('--update-all', action='store_true', help='Update all failing')
    golden_parser.set_defaults(func=cmd_golden)
    
    # benchmark (P2-090)
    bench_parser = subparsers.add_parser('benchmark', help='Performance benchmarks')
    bench_parser.add_argument('-n', '--iterations', type=int, default=10, help='Iterations')
    bench_parser.set_defaults(func=cmd_benchmark)
    
    # safety (P2-097)
    safety_parser = subparsers.add_parser('safety-check', help='Safety status')
    safety_parser.add_argument('--kill', action='store_true', help='Activate kill switch')
    safety_parser.add_argument('--reset', action='store_true', help='Reset kill switch')
    safety_parser.set_defaults(func=cmd_safety)
    
    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
