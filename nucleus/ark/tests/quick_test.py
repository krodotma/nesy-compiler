#!/usr/bin/env python3
"""
quick_test.py - Quick operational verification of ARK components
"""

import sys
sys.path.insert(0, '/Users/kroma/pluribus')

print("=" * 60)
print("ARK OPERATIONAL ENTELECHEIA PROOF")
print("=" * 60)

# Test 1: Imports
print("\n[TEST 1] Module Imports")
try:
    from nucleus.ark.core.repository import ArkRepository
    from nucleus.ark.core.context import ArkCommitContext
    from nucleus.ark.gates.inertia import InertiaGate, InertiaContext
    from nucleus.ark.gates.entelecheia import EntelecheiaGate, EntelecheiaContext
    from nucleus.ark.gates.homeostasis import HomeostasisGate, HomeostasisContext
    from nucleus.ark.ribosome.gene import Gene
    from nucleus.ark.ribosome.clade import Clade
    from nucleus.ark.ribosome.genome import OrganismGenome
    from nucleus.ark.rhizom.dag import RhizomDAG, RhizomNode
    from nucleus.ark.rhizom.etymology import EtymologyExtractor
    from nucleus.ark.rhizom.lineage import LineageTracker
    from nucleus.ark.portal.ingest import IngestPipeline
    from nucleus.ark.portal.distill import DistillationPipeline
    from nucleus.ark.synthesis.ltl_spec import PluribusLTLSpec, LTLVerifier
    from nucleus.ark.synthesis.grammar import GrammarFilter, SynthesisGrammar
    print("✅ All 16 ARK modules imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: DNA Gates
print("\n[TEST 2] DNA Gates Verification")

# Inertia Gate
gate = InertiaGate()
ctx = InertiaContext(files=['some_file.py'], has_witness=False)
result = gate.check(ctx)
print(f"  InertiaGate (normal file, no witness): {result}")
assert result == True, "Normal file should pass"

ctx2 = InertiaContext(files=['world_router.py'], has_witness=False)
result2 = gate.check(ctx2)
print(f"  InertiaGate (core file, no witness): {result2}")
assert result2 == False, "Core file without witness should fail"

ctx3 = InertiaContext(files=['world_router.py'], has_witness=True)
result3 = gate.check(ctx3)
print(f"  InertiaGate (core file, with witness): {result3}")
assert result3 == True, "Core file with witness should pass"
print("✅ InertiaGate works correctly")

# Entelecheia Gate
egate = EntelecheiaGate()
ectx = EntelecheiaContext(purpose='Implement user auth', is_cosmetic=False)
eresult = egate.check(ectx)
print(f"  EntelecheiaGate (purposeful): {eresult}")
assert eresult == True, "Purposeful change should pass"

ectx2 = EntelecheiaContext(purpose='formatting', is_cosmetic=True)
eresult2 = egate.check(ectx2)
print(f"  EntelecheiaGate (cosmetic only): {eresult2}")
assert eresult2 == False, "Cosmetic without spec should fail"
print("✅ EntelecheiaGate works correctly")

# Homeostasis Gate
hgate = HomeostasisGate()
hctx = HomeostasisContext(entropy={'h_struct': 0.3, 'h_doc': 0.2})
hresult = hgate.check(hctx)
print(f"  HomeostasisGate (low entropy): {hresult}")
assert hresult == True, "Low entropy should pass"

hctx2 = HomeostasisContext(entropy={'h_struct': 0.9, 'h_doc': 0.9}, threshold=0.7)
hresult2 = hgate.check(hctx2)
print(f"  HomeostasisGate (high entropy): {hresult2}")
assert hresult2 == False, "High entropy should fail"
print("✅ HomeostasisGate works correctly")

# Test 3: Etymology
print("\n[TEST 3] Rhizome Etymology")
extractor = EtymologyExtractor()
code = '"""User authentication module."""\ndef authenticate(user, password): pass'
etymology = extractor.extract_from_code(code, 'auth.py')
print(f"  Primary: {etymology.primary}")
print(f"  Keywords: {etymology.keywords[:5]}")
print(f"  Domain: {etymology.domain}")
assert len(etymology.primary) > 0, "Should extract primary"
assert len(etymology.keywords) > 0, "Should extract keywords"
print("✅ EtymologyExtractor works correctly")

# Test 4: LTL Verification
print("\n[TEST 4] LTL Specification")
spec = PluribusLTLSpec.core_spec()
print(f"  Spec loaded: {spec.name}")
print(f"  Safety formulas: {len(spec.safety_formulas)}")
print(f"  Liveness formulas: {len(spec.liveness_formulas)}")
assert len(spec.formulas) > 0, "Should have formulas"

verifier = LTLVerifier(spec)
trace = [{'commit': True, 'inertia_pass': True, 'entelecheia_pass': True, 
          'homeostasis_pass': True, 'stable': True}]
verified = verifier.verify_trace(trace)
print(f"  Compliant trace verified: {verified}")
assert verified == True, "Compliant trace should pass"

bad_trace = [{'commit': True, 'inertia_pass': False}]
bad_verified = verifier.verify_trace(bad_trace)
print(f"  Violating trace detected: {not bad_verified}")
assert bad_verified == False, "Violating trace should fail"
print("✅ LTL Verifier works correctly")

# Test 5: Grammar Filter
print("\n[TEST 5] Grammar Filter")
grammar = SynthesisGrammar.strict()
gfilter = GrammarFilter(grammar)

good_code = 'def simple_function():\n    return 42'
valid, violations = gfilter.validate(good_code)
print(f"  Simple function valid: {valid}")
assert valid == True, "Simple function should pass"

bad_code = 'class AbstractManagerFactory:\n    pass'
valid2, violations2 = gfilter.validate(bad_code)
print(f"  Anti-pattern valid: {valid2}")
print(f"  Violations: {violations2}")
assert valid2 == False, "Anti-pattern should fail"
print("✅ GrammarFilter works correctly")

# Test 6: Thompson Sampling
print("\n[TEST 6] Thompson Sampling")
clade_a = Clade(name='high-performer', alpha=20, beta=2)
clade_b = Clade(name='low-performer', alpha=2, beta=20)

print(f"  Clade A expected fitness: {clade_a.expected_fitness():.2f}")
print(f"  Clade B expected fitness: {clade_b.expected_fitness():.2f}")
assert clade_a.expected_fitness() > 0.8, "High performer should have high fitness"
assert clade_b.expected_fitness() < 0.2, "Low performer should have low fitness"

a_wins = sum(1 for _ in range(100) if clade_a.sample_fitness() > clade_b.sample_fitness())
print(f"  High performer selected: {a_wins}/100 times")
assert a_wins > 80, "High performer should win most samples"
print("✅ Thompson Sampling works correctly")

# Test 7: Gene Creation
print("\n[TEST 7] Gene Creation")
gene = Gene.from_file('test.py', '"""Test."""\ndef foo(): pass\n', 'Test gene')
print(f"  Gene OID: {gene.oid}")
print(f"  Gene lines: {gene.lines}")
print(f"  Gene inertia: {gene.calculate_inertia():.2f}")
assert gene.oid != "", "Should have OID"
assert gene.lines > 0, "Should count lines"
print("✅ Gene creation works correctly")

# Test 8: Rhizome DAG
print("\n[TEST 8] Rhizome DAG")
import tempfile
import shutil
test_dir = tempfile.mkdtemp(prefix="rhizome_test_")
try:
    from pathlib import Path
    rhizom = RhizomDAG(Path(test_dir))
    
    # Insert nodes
    rhizom.insert(RhizomNode(sha="abc123", etymology="Auth module", cmp=0.7))
    rhizom.insert(RhizomNode(sha="def456", etymology="Database layer", cmp=0.8))
    rhizom.insert(RhizomNode(sha="ghi789", etymology="Auth token validation", cmp=0.75))
    
    # Query
    auth_nodes = rhizom.query_by_etymology("auth")
    print(f"  Inserted 3 nodes")
    print(f"  Query 'auth' found: {len(auth_nodes)} nodes")
    assert len(auth_nodes) == 2, "Should find 2 auth-related nodes"
    
    # Retrieve
    node = rhizom.get("abc123")
    print(f"  Retrieved node etymology: {node.etymology}")
    assert node.etymology == "Auth module", "Should retrieve correct node"
    print("✅ Rhizome DAG works correctly")
finally:
    shutil.rmtree(test_dir, ignore_errors=True)

print("\n" + "=" * 60)
print("🎉 ALL 8 OPERATIONAL TESTS PASSED - ENTELECHEIA PROVEN")
print("=" * 60)
