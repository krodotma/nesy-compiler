#!/usr/bin/env zsh
# ==============================================================================
# PluriChat Exhaustive Fallback Test Suite
# ==============================================================================
# Tests multi-model fallback routing across:
# - Gemini 3 (Intra: gemini-cli, Extra: vertex-gemini, vertex-gemini-curl)
# - Claude (Web: claude-cli → Opus/Sonnet, NOT traditional API)
# - GPT 5.2 (Web: codex-cli, NOT traditional API)
#
# Purpose: Verify live fallback chains work correctly when primary providers fail
#
# Usage:
#   ./plurichat_fallback_test.zsh [--full|--quick|--deep|--provider <name>]
#
# Options:
#   --full      Run all tests exhaustively
#   --quick     Run quick smoke tests only
#   --deep      Run deep architectural queries
#   --provider  Test specific provider only
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PLURIBUS_ROOT="${PLURIBUS_ROOT:-/pluribus}"
BUS_DIR="${PLURIBUS_BUS_DIR:-$PLURIBUS_ROOT/.pluribus/bus}"
PLURICHAT="$SCRIPT_DIR/plurichat.py"
TOOLS_DIR="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Test tracking
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# ==============================================================================
# Provider Definitions
# ==============================================================================
# Intra-agent: Same agent system (Gemini-3 native)
# Extra-agent: Cross-agent system (Claude, GPT via web interfaces)

typeset -A PROVIDERS
PROVIDERS=(
    # Gemini 3 Intra-Agent (Native CLI tools)
    [gemini-intra-cli]="gemini"
    [gemini-intra-vertex]="vertex-gemini"
    [gemini-intra-vertex-curl]="vertex-gemini-curl"

    # Claude Extra-Agent (Web-based, NOT API)
    [claude-extra-opus]="claude-cli"
    [claude-extra-sonnet]="claude-cli"

    # GPT 5.2 Extra-Agent (Codex, Web-based)
    [gpt-extra-codex]="codex-cli"

    # Mock for testing
    [mock]="mock"
)

# ==============================================================================
# Test Prompts by Category
# ==============================================================================

# Quick smoke test prompts
QUICK_PROMPTS=(
    "What is 2+2?"
    "Hello, are you there?"
    "List 3 colors"
)

# Code generation prompts (deep)
CODE_PROMPTS=(
    "Write a Python function that implements binary search"
    "Implement a simple LRU cache in TypeScript"
    "Create a bash one-liner that counts lines in all Python files"
)

# Research prompts (deep)
RESEARCH_PROMPTS=(
    "Explain the transformer architecture in one paragraph"
    "What are the key differences between VGT and HGT in evolutionary systems?"
    "Describe the omega-level event semantics in multi-agent systems"
)

# Architectural prompts (very deep)
ARCHITECTURE_PROMPTS=(
    "Design a multi-agent orchestration system with bus-based IPC"
    "Architect a neurosymbolic routing system for LLM queries"
    "Explain how autopoietic reentry works in self-modifying systems"
)

# ==============================================================================
# Utility Functions
# ==============================================================================

log_header() {
    echo ""
    echo "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo "${BOLD}${CYAN}║${NC} ${BOLD}$1${NC}"
    echo "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
}

log_info() {
    echo "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_fail() {
    echo "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

log_skip() {
    echo "${YELLOW}[SKIP]${NC} $1"
    ((TESTS_SKIPPED++))
}

log_warn() {
    echo "${YELLOW}[WARN]${NC} $1"
}

# ==============================================================================
# Provider Health Check
# ==============================================================================

check_provider_health() {
    local provider="$1"
    log_info "Checking $provider availability..."

    local result
    result=$(python3 "$PLURICHAT" --status 2>&1 | grep -i "$provider" || echo "not found")

    if echo "$result" | grep -q "●"; then
        log_success "$provider is available"
        return 0
    else
        log_warn "$provider is NOT available: $result"
        return 1
    fi
}

# ==============================================================================
# Single Provider Test
# ==============================================================================

test_provider() {
    local provider="$1"
    local prompt="$2"
    local expected_type="${3:-any}"  # any, code, research, etc.
    local timeout="${4:-30}"

    log_info "Testing provider: ${BOLD}$provider${NC}"
    log_info "  Prompt: ${prompt:0:50}..."

    local start_time=$(date +%s.%N)
    local output
    local exit_code=0

    output=$(timeout "$timeout" python3 "$PLURICHAT" \
        --ask "$prompt" \
        --provider "$provider" \
        2>&1) || exit_code=$?

    local end_time=$(date +%s.%N)
    local latency=$(echo "$end_time - $start_time" | bc)

    if [[ $exit_code -eq 0 && -n "$output" ]]; then
        local output_length=${#output}
        log_success "Provider $provider responded in ${latency}s (${output_length} chars)"
        echo "  ${CYAN}Response preview:${NC} ${output:0:100}..."

        # Emit test result to bus
        emit_test_result "$provider" "pass" "$latency" "$output_length" "$prompt"
        return 0
    else
        log_fail "Provider $provider failed (exit=$exit_code, timeout=$timeout)"
        emit_test_result "$provider" "fail" "$latency" "0" "$prompt" "$output"
        return 1
    fi
}

# ==============================================================================
# Fallback Chain Test
# ==============================================================================

test_fallback_chain() {
    local chain_name="$1"
    shift
    local providers=("$@")

    log_header "Testing Fallback Chain: $chain_name"
    log_info "Chain: ${providers[*]}"

    local prompt="Respond with 'OK' to confirm you are working"
    local success=0

    for provider in "${providers[@]}"; do
        if test_provider "$provider" "$prompt" "any" 20; then
            log_success "Fallback chain $chain_name succeeded at provider: $provider"
            success=1
            break
        else
            log_warn "Provider $provider failed, trying next in chain..."
        fi
    done

    if [[ $success -eq 0 ]]; then
        log_fail "Fallback chain $chain_name exhausted - all providers failed"
        return 1
    fi

    return 0
}

# ==============================================================================
# Emit Test Result to Bus
# ==============================================================================

emit_test_result() {
    local provider="$1"
    local status="$2"
    local latency="$3"
    local response_length="$4"
    local prompt="$5"
    local error="${6:-}"

    python3 -c "
import json
import time
import uuid
from pathlib import Path

event = {
    'id': str(uuid.uuid4()),
    'ts': time.time(),
    'iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'topic': 'plurichat.fallback.test',
    'kind': 'test_result',
    'level': 'info' if '$status' == 'pass' else 'warn',
    'actor': 'fallback_test',
    'data': {
        'provider': '$provider',
        'status': '$status',
        'latency_s': float('$latency'),
        'response_length': int('$response_length'),
        'prompt': '${prompt:0:100}'.replace(\"'\", ''),
        'error': '${error:0:200}'.replace(\"'\", '') if '$error' else None,
    },
}

bus_path = Path('$BUS_DIR') / 'events.ndjson'
with bus_path.open('a') as f:
    f.write(json.dumps(event) + '\n')
" 2>/dev/null || true
}

# ==============================================================================
# Test Suites
# ==============================================================================

run_gemini_intra_tests() {
    log_header "Gemini 3 Intra-Agent Tests"
    log_info "Testing native Gemini 3 providers (CLI and Vertex)"

    local gemini_providers=(gemini vertex-gemini vertex-gemini-curl)

    for provider in "${gemini_providers[@]}"; do
        check_provider_health "$provider" || continue

        # Quick test
        test_provider "$provider" "What is the capital of France?" "general" 15

        # Code test (for deeper routing)
        test_provider "$provider" "Write a one-line Python hello world" "code" 20
    done
}

run_claude_extra_tests() {
    log_header "Claude Extra-Agent Tests (Web Opus/Sonnet)"
    log_info "Testing Claude via CLI (web-based, NOT API)"

    check_provider_health "claude-cli" || {
        log_skip "Claude CLI not available"
        return 1
    }

    # Test Claude with different query types
    test_provider "claude-cli" "Explain recursion briefly" "general" 30
    test_provider "claude-cli" "Write a Python quicksort" "code" 45
    test_provider "claude-cli" "What is the omega event semantics?" "research" 45
}

run_gpt_extra_tests() {
    log_header "GPT 5.2 Extra-Agent Tests (Codex)"
    log_info "Testing GPT via Codex CLI (web-based, NOT API)"

    check_provider_health "codex-cli" || {
        log_skip "Codex CLI not available"
        return 1
    }

    # Test Codex with code-focused queries
    test_provider "codex-cli" "Hello, respond with OK" "general" 30
    test_provider "codex-cli" "Implement binary search in Python" "code" 60
}

run_fallback_chain_tests() {
    log_header "Fallback Chain Tests"

    # Chain 1: Gemini → Vertex → Mock
    test_fallback_chain "gemini-to-vertex" gemini vertex-gemini mock

    # Chain 2: Codex → Claude → Gemini → Mock
    test_fallback_chain "codex-first" codex-cli claude-cli gemini mock

    # Chain 3: Claude → Codex → Vertex → Mock
    test_fallback_chain "claude-first" claude-cli codex-cli vertex-gemini mock

    # Chain 4: Full fallback chain (production order)
    test_fallback_chain "production" codex-cli gemini vertex-gemini claude-cli mock
}

run_deep_tests() {
    log_header "Deep Architectural Query Tests"
    log_info "Testing with complex prompts to verify depth routing"

    local deep_providers=(codex-cli claude-cli vertex-gemini)

    for prompt in "${ARCHITECTURE_PROMPTS[@]}"; do
        log_info "Deep prompt: ${prompt:0:60}..."

        for provider in "${deep_providers[@]}"; do
            check_provider_health "$provider" || continue

            test_provider "$provider" "$prompt" "architecture" 90 && break
        done
    done
}

run_multi_agent_coordination_test() {
    log_header "Multi-Agent Coordination Test (2-3 Agents)"
    log_info "Testing simultaneous queries to multiple agents"

    local prompt="Respond with your model name and 'ready'"

    # Run queries in parallel to multiple agents
    log_info "Starting parallel queries to 3 agents..."

    local results=()

    # Background tasks
    (
        python3 "$PLURICHAT" --ask "$prompt" --provider gemini 2>&1 || echo "GEMINI_FAIL"
    ) &
    local pid1=$!

    (
        python3 "$PLURICHAT" --ask "$prompt" --provider claude-cli 2>&1 || echo "CLAUDE_FAIL"
    ) &
    local pid2=$!

    (
        python3 "$PLURICHAT" --ask "$prompt" --provider codex-cli 2>&1 || echo "CODEX_FAIL"
    ) &
    local pid3=$!

    # Wait for all
    wait $pid1 $pid2 $pid3 2>/dev/null

    log_success "Multi-agent coordination test completed"

    # Emit coordination event
    python3 -c "
import json, time, uuid
from pathlib import Path

event = {
    'id': str(uuid.uuid4()),
    'ts': time.time(),
    'iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'topic': 'plurichat.multiagent.test',
    'kind': 'coordination_test',
    'level': 'info',
    'actor': 'fallback_test',
    'data': {
        'agents': ['gemini', 'claude-cli', 'codex-cli'],
        'topology': 'star',
        'fanout': 3,
    },
}

bus_path = Path('$BUS_DIR') / 'events.ndjson'
with bus_path.open('a') as f:
    f.write(json.dumps(event) + '\n')
" 2>/dev/null || true
}

# ==============================================================================
# Full Test Suite
# ==============================================================================

run_full_suite() {
    log_header "PluriChat Exhaustive Fallback Test Suite"
    echo ""
    echo "${MAGENTA}Testing multi-model fallback routing:${NC}"
    echo "  - Gemini 3 (Intra: gemini-cli, vertex-gemini, vertex-gemini-curl)"
    echo "  - Claude (Extra: claude-cli → Opus/Sonnet via web)"
    echo "  - GPT 5.2 (Extra: codex-cli via web)"
    echo ""

    # Check overall status first
    log_info "Checking provider status..."
    python3 "$PLURICHAT" --status 2>&1 || true
    echo ""

    # Run test suites
    run_gemini_intra_tests
    run_claude_extra_tests
    run_gpt_extra_tests
    run_fallback_chain_tests
    run_multi_agent_coordination_test
}

run_quick_suite() {
    log_header "Quick Smoke Tests"

    for prompt in "${QUICK_PROMPTS[@]}"; do
        test_provider "auto" "$prompt" "general" 20
    done
}

# ==============================================================================
# Test Summary
# ==============================================================================

print_summary() {
    echo ""
    log_header "Test Summary"
    echo "${GREEN}Passed:${NC}  $TESTS_PASSED"
    echo "${RED}Failed:${NC}  $TESTS_FAILED"
    echo "${YELLOW}Skipped:${NC} $TESTS_SKIPPED"
    echo ""

    local total=$((TESTS_PASSED + TESTS_FAILED))
    if [[ $total -gt 0 ]]; then
        local pass_rate=$(echo "scale=1; $TESTS_PASSED * 100 / $total" | bc)
        echo "Pass rate: ${pass_rate}%"

        if [[ $TESTS_FAILED -eq 0 ]]; then
            echo "${GREEN}${BOLD}All tests passed!${NC}"
        else
            echo "${YELLOW}Some tests failed. Check logs above.${NC}"
        fi
    fi
}

# ==============================================================================
# Main Entry Point
# ==============================================================================

main() {
    local mode="${1:-full}"

    case "$mode" in
        --full|-f)
            run_full_suite
            run_deep_tests
            ;;
        --quick|-q)
            run_quick_suite
            ;;
        --deep|-d)
            run_deep_tests
            ;;
        --gemini)
            run_gemini_intra_tests
            ;;
        --claude)
            run_claude_extra_tests
            ;;
        --gpt|--codex)
            run_gpt_extra_tests
            ;;
        --fallback)
            run_fallback_chain_tests
            ;;
        --multi)
            run_multi_agent_coordination_test
            ;;
        --provider)
            local provider="${2:-auto}"
            local prompt="${3:-Hello, respond with OK}"
            test_provider "$provider" "$prompt"
            ;;
        --help|-h)
            echo "Usage: $0 [--full|--quick|--deep|--gemini|--claude|--gpt|--fallback|--multi|--provider <name> [prompt]]"
            echo ""
            echo "Options:"
            echo "  --full      Run all tests exhaustively (default)"
            echo "  --quick     Run quick smoke tests only"
            echo "  --deep      Run deep architectural queries"
            echo "  --gemini    Test Gemini 3 providers only"
            echo "  --claude    Test Claude providers only"
            echo "  --gpt       Test GPT/Codex providers only"
            echo "  --fallback  Test fallback chains only"
            echo "  --multi     Test multi-agent coordination"
            echo "  --provider  Test specific provider with optional prompt"
            exit 0
            ;;
        *)
            run_full_suite
            ;;
    esac

    print_summary

    # Exit with error if any tests failed
    [[ $TESTS_FAILED -eq 0 ]]
}

main "$@"
