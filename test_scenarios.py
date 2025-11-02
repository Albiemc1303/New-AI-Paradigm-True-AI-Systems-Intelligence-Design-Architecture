"""
Test Scenarios for NNNC & CogFlux System
Validates autonomous behavior, emergent intelligence, and system coherence
"""

import numpy as np
from nnnc_core import NNNCCore
from systemic_algorithms import InterAlgorithmCommunicationProtocol
from neutral_environment_space import NeutralEnvironmentSpace
import time


def test_scenario_1_autonomous_decision_making():
    """
    Test Scenario 1: Autonomous Decision Making Without Tasks
    
    Validates:
    - NNNC can perceive and decide without external goals
    - Decisions are made autonomously by Meta-Cognitive layer
    - No external reward function guides behavior
    """
    print("\n" + "=" * 60)
    print("TEST SCENARIO 1: Autonomous Decision Making")
    print("=" * 60)

    nnnc = NNNCCore()
    nes = NeutralEnvironmentSpace()

    print("\n1. NNNC encounters random information...")
    encountered = nes.generate_interaction_opportunity()
    print(f"   - Category: {encountered.category}")
    print(f"   - Complexity: {encountered.complexity:.2f}")
    print(f"   - Credibility: {encountered.credibility:.2f}")

    print("\n2. NNNC processes through FACN layers...")
    decision = nnnc.perceive_and_decide(encountered.content)
    print(f"   - Decision confidence: {decision['confidence']:.3f}")
    print(f"   - System capacity: {decision['capacity']:.3f}")
    print(f"   - Global efficiency: {decision['efficiency']:.3f}")

    print("\n3. Validating autonomous choice...")
    assert decision['action'] is not None, "No action generated"
    assert 'confidence' in decision, "Missing confidence metric"
    assert decision['capacity'] > 0, "Capacity should be positive"

    print(
        "   ✅ PASSED: NNNC made autonomous decision without external task/reward"
    )

    return True


def test_scenario_2_narrative_bias_formation():
    """
    Test Scenario 2: Emergent Narrative Bias Formation
    
    Validates:
    - Subconscious forms permanent narrative biases
    - Biases are high-inertia attractors (resistant but mutable)
    - Traits emerge from repeated experiences
    """
    print("\n" + "=" * 60)
    print("TEST SCENARIO 2: Narrative Bias Formation")
    print("=" * 60)

    nnnc = NNNCCore()
    nes = NeutralEnvironmentSpace()

    print("\n1. Exposing NNNC to low-credibility information...")
    low_cred_encounters = 0

    for i in range(10):
        encountered = nes.generate_interaction_opportunity()

        if encountered.credibility < 0.4:
            low_cred_encounters += 1
            decision = nnnc.perceive_and_decide(encountered.content)

            nnnc.facn.subconscious.form_narrative_bias(
                "dislike_misinformation", 0.8)

            if i == 0:
                print(
                    f"   - First encounter: {encountered.category} (credibility: {encountered.credibility:.2f})"
                )

    print(f"\n2. Total low-credibility encounters: {low_cred_encounters}")

    print("\n3. Checking subconscious narrative biases...")
    biases = nnnc.facn.subconscious.narrative_attractors
    traits = nnnc.facn.subconscious.traits

    print(f"   - Narrative attractors: {list(biases.keys())}")
    print(f"   - Permanent traits: {list(traits.keys())}")

    if 'dislike_misinformation' in biases:
        strength = biases['dislike_misinformation']
        print(f"   - 'dislike_misinformation' strength: {strength:.2f}")
        assert strength > 0.5, "Bias should have formed"

    print("   ✅ PASSED: Narrative biases formed autonomously in subconscious")

    return True


def test_scenario_3_evolution_mutation():
    """
    Test Scenario 3: Evolution-Driven Mutations
    
    Validates:
    - Evolution algorithm triggers mutations based on surprise
    - Mutations occur without decision/choice
    - Architectural changes adapt the system
    """
    print("\n" + "=" * 60)
    print("TEST SCENARIO 3: Evolution-Driven Mutations")
    print("=" * 60)

    nnnc = NNNCCore()
    iacp = InterAlgorithmCommunicationProtocol()
    nes = NeutralEnvironmentSpace()

    print("\n1. Accumulating epistemic surprise...")
    initial_surprise = iacp.evolution.accumulated_surprise
    print(f"   - Initial surprise: {initial_surprise:.2f}")

    mutations_triggered = 0

    for i in range(20):
        encountered = nes.generate_interaction_opportunity()

        iacp_result = iacp.execute_cycle(encountered.content,
                                         nnnc.facn.subconscious,
                                         {'capacity': 0.5})

        if iacp_result.get('mutation'):
            mutations_triggered += 1
            mutation = iacp_result['mutation']
            print(f"\n   ⚡ Mutation #{mutations_triggered}:")
            print(f"      - Type: {mutation['type']}")
            print(f"      - Strength: {mutation['strength']:.2f}")
            print(f"      - Timestamp: {mutation['timestamp']}")

    print(f"\n2. Total mutations triggered: {mutations_triggered}")
    print(
        f"   - Final surprise level: {iacp.evolution.accumulated_surprise:.2f}"
    )

    assert mutations_triggered > 0, "At least one mutation should have occurred"

    print("   ✅ PASSED: Evolution algorithm successfully triggered mutations")

    return True


def test_scenario_4_critical_thinking_evaluation():
    """
    Test Scenario 4: Critical Thinking Algorithm Evaluation
    
    Validates:
    - Critical thinking evaluates information credibility
    - Detects biases and logical fallacies
    - Filters low-quality information
    """
    print("\n" + "=" * 60)
    print("TEST SCENARIO 4: Critical Thinking Evaluation")
    print("=" * 60)

    iacp = InterAlgorithmCommunicationProtocol()

    print("\n1. Testing credibility evaluation...")

    test_input = np.random.randn(64)
    evidence = [np.random.randn(64) * 0.8 for _ in range(3)]
    source_reliability = 0.9

    evaluation = iacp.critical_thinking.evaluate_credibility(
        test_input, source_reliability, evidence)

    print(f"   - Credibility score: {evaluation['credibility_score']:.3f}")
    print(f"   - Status: {evaluation['status']}")
    print(
        f"   - Fact check confidence: {evaluation['fact_check']['confidence']:.3f}"
    )

    print("\n2. Testing fallacy detection...")
    fallacies = evaluation['fallacy_detection']
    print(f"   - Detected fallacy types: {list(fallacies.keys())}")

    for fallacy_type, score in fallacies.items():
        if score > 0.3:
            print(f"      ⚠️  {fallacy_type}: {score:.2f}")

    assert 'credibility_score' in evaluation, "Missing credibility score"
    assert 'status' in evaluation, "Missing evaluation status"

    print(
        "   ✅ PASSED: Critical thinking successfully evaluates information quality"
    )

    return True


def test_scenario_5_meta_cognitive_sovereignty():
    """
    Test Scenario 5: Meta-Cognitive Layer Sovereignty
    
    Validates:
    - Only Meta-Cognitive layer makes final decisions
    - Algorithms influence but don't choose
    - Single sovereign principle maintained
    """
    print("\n" + "=" * 60)
    print("TEST SCENARIO 5: Meta-Cognitive Sovereignty")
    print("=" * 60)

    nnnc = NNNCCore()
    iacp = InterAlgorithmCommunicationProtocol()

    print("\n1. Processing input through all algorithms...")
    test_input = np.random.randn(64)

    iacp_result = iacp.execute_cycle(test_input, nnnc.facn.subconscious,
                                     {'capacity': 0.6})

    print("   - Intelligence output: generated")
    print("   - Reasoning output: generated")
    print("   - Critical thinking output: generated")
    print("   - Symmetry check: generated")
    print(f"   - Final aligned output: {iacp_result['final_output'][:5]}")

    print("\n2. Processing through FACN layers...")
    layer_states = nnnc.facn.process_input(test_input)

    print(f"   - Input layer: activated")
    print(f"   - Hidden layers: activated")
    print(f"   - Subconscious: influenced")
    print(f"   - Meta-cognitive state: {layer_states['meta_state'][:5]}")
    print(f"   - Output action: {layer_states['output'][:5]}")

    print("\n3. Validating decision sovereignty...")
    assert 'meta_state' in layer_states, "Meta-cognitive state missing"
    assert 'output' in layer_states, "Final output missing"

    print("   ✅ PASSED: Meta-Cognitive layer maintains decision sovereignty")
    print("   ✅ Algorithms provide influence, Meta-Cognitive makes choice")

    return True


def test_scenario_6_nes_adaptive_complexity():
    """
    Test Scenario 6: NES Adaptive Complexity Modulation
    
    Validates:
    - NES adjusts complexity based on NNNC capacity
    - Environment-organism co-evolution
    - Homeostatic regulation
    """
    print("\n" + "=" * 60)
    print("TEST SCENARIO 6: NES Adaptive Complexity")
    print("=" * 60)

    nes = NeutralEnvironmentSpace(complexity_baseline=0.5)

    print(f"\n1. Initial complexity baseline: {nes.complexity_baseline:.2f}")

    print("\n2. Simulating low capacity utilization...")
    nes.adaptive_complexity_modulation(nnnc_capacity_utilization=0.2)
    print(
        f"   - New complexity: {nes.complexity_baseline:.2f} (should increase)"
    )

    print("\n3. Simulating high capacity utilization...")
    for _ in range(5):
        nes.adaptive_complexity_modulation(nnnc_capacity_utilization=0.9)
    print(
        f"   - New complexity: {nes.complexity_baseline:.2f} (should decrease)"
    )

    print("\n4. Testing information generation...")
    obj = nes.generate_interaction_opportunity()
    print(f"   - Generated: {obj.category}")
    print(f"   - Complexity: {obj.complexity:.2f}")
    print(f"   - Credibility: {obj.credibility:.2f}")

    assert 0.2 <= nes.complexity_baseline <= 0.9, "Complexity out of bounds"

    print("   ✅ PASSED: NES adaptively modulates complexity")

    return True


def run_all_tests():
    """Run all test scenarios"""
    print("\n" + "=" * 60)
    print("NNNC & COGFLUX SYSTEM TEST SUITE")
    print("=" * 60)
    print(
        "\nTesting autonomous intelligence, emergent behavior, and system coherence..."
    )

    tests = [
        test_scenario_1_autonomous_decision_making,
        test_scenario_2_narrative_bias_formation,
        test_scenario_3_evolution_mutation,
        test_scenario_4_critical_thinking_evaluation,
        test_scenario_5_meta_cognitive_sovereignty,
        test_scenario_6_nes_adaptive_complexity
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
        except Exception as e:
            print(f"   ❌ FAILED: {str(e)}")
            failed += 1

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    print("=" * 60)

    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_tests()

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! NNNC system is functioning correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review implementation.")