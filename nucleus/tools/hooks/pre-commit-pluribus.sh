#!/bin/bash
#
# pre-commit-pluribus.sh - Git pre-commit hook for Pluribus agent identity enforcement
#
# INSTALL:
#   ln -sf ../../pluribus_next/tools/hooks/pre-commit-pluribus.sh .git/hooks/pre-commit
#
# PURPOSE:
#   1. Enforces PLURIBUS_ACTOR identity for proper commit attribution
#   2. Warns about git CLI usage (should use iso_git.mjs)
#   3. Runs dashboard typecheck for dashboard files
#   4. Emits bus evidence for audit trail
#
# AGENTS MUST:
#   export PLURIBUS_ACTOR="claude-opus"  # or codex, gemini, etc.
#   Use: node pluribus_next/tools/iso_git.mjs commit ...
#   NOT: git commit ...
#

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
BUS_DIR="${PLURIBUS_BUS_DIR:-$REPO_ROOT/.pluribus/bus}"
EVENTS_FILE="$BUS_DIR/events.ndjson"

# --- Helper: Emit bus event ---
emit_bus_event() {
    local topic="$1"
    local kind="$2"
    local data="$3"

    if [ -d "$BUS_DIR" ] && [ -w "$EVENTS_FILE" ] 2>/dev/null; then
        local ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        local actor="${PLURIBUS_ACTOR:-unknown}"
        echo "{\"ts\":\"$ts\",\"topic\":\"$topic\",\"kind\":\"$kind\",\"actor\":\"$actor\",\"data\":$data}" >> "$EVENTS_FILE"
    fi
}

# --- Guard 1: PLURIBUS_ACTOR identity check ---
echo "=========================================="
echo "  PLURIBUS GUARD: Identity & Provenance"
echo "=========================================="

if [ -z "$PLURIBUS_ACTOR" ]; then
    echo ""
    echo "  ⚠️  WARNING: PLURIBUS_ACTOR not set!"
    echo ""
    echo "  Your commit will be attributed incorrectly."
    echo "  Set your identity before committing:"
    echo ""
    echo "    export PLURIBUS_ACTOR=\"claude-opus\"   # or codex, gemini, etc."
    echo "    export PLURIBUS_GIT_EMAIL=\"claude@pluribus.local\""
    echo ""
    echo "  Recommended: Use iso_git.mjs instead of git CLI:"
    echo "    node pluribus_next/tools/iso_git.mjs commit /pluribus \"...\""
    echo ""

    # Emit violation event
    emit_bus_event "git.identity.warning" "metric" "{\"error\":\"PLURIBUS_ACTOR not set\",\"caller\":\"pre-commit\"}"

    # ENFORCEMENT MODE - Block if PLURIBUS_ENFORCE_IDENTITY=1 (default: warn only)
    if [ "${PLURIBUS_ENFORCE_IDENTITY:-0}" = "1" ]; then
        echo "=========================================="
        echo "  COMMIT BLOCKED - Set PLURIBUS_ACTOR"
        echo "=========================================="
        exit 1
    else
        echo "  Proceeding with warning (identity may be incorrect)..."
        echo ""
    fi
else
    echo "  ✓ PLURIBUS_ACTOR: $PLURIBUS_ACTOR"

    # Check if git config identity matches
    GIT_NAME=$(git config user.name 2>/dev/null || echo "")
    GIT_EMAIL=$(git config user.email 2>/dev/null || echo "")

    if [ -n "$GIT_NAME" ] && [ "$GIT_NAME" != "$PLURIBUS_ACTOR" ]; then
        echo "  ⚠️  Git config user.name ($GIT_NAME) differs from PLURIBUS_ACTOR"
        echo "     Consider: git config --unset user.name"
    fi

    emit_bus_event "git.commit.start" "metric" "{\"actor\":\"$PLURIBUS_ACTOR\",\"via\":\"git-cli\"}"
fi

# --- Guard 2: Detect git CLI vs iso_git usage ---
# Check if this commit was initiated via iso_git (it sets a marker env var)
if [ -z "$PLURIBUS_ISO_GIT" ]; then
    echo ""
    echo "  ⚠️  Direct git CLI detected (not iso_git.mjs)"
    echo ""
    echo "  Per AGENTS.md, agents should use iso_git.mjs for:"
    echo "    - Proper identity attribution"
    echo "    - PQC signatures (ML-DSA-65)"
    echo "    - HGT guard ladder checks"
    echo ""
    echo "  Command:"
    echo "    PLURIBUS_ACTOR=$PLURIBUS_ACTOR node pluribus_next/tools/iso_git.mjs commit /pluribus \"...\""
    echo ""

    emit_bus_event "git.cli.warning" "metric" "{\"reason\":\"should use iso_git.mjs\",\"actor\":\"${PLURIBUS_ACTOR:-unknown}\"}"

    # ENFORCEMENT MODE - Block if PLURIBUS_ENFORCE_ISO_GIT=1 (default: warn only)
    if [ "${PLURIBUS_ENFORCE_ISO_GIT:-0}" = "1" ]; then
        echo "=========================================="
        echo "  COMMIT BLOCKED - Use iso_git.mjs"
        echo "=========================================="
        echo "  Run: PLURIBUS_ACTOR=$PLURIBUS_ACTOR node pluribus_next/tools/iso_git.mjs commit ..."
        echo ""
        exit 1
    fi
fi

echo "=========================================="
echo ""

# --- Guard 3: Dashboard typecheck (existing functionality) ---
DASHBOARD_DIR="$REPO_ROOT/pluribus_next/dashboard"

# Check if any dashboard files are staged
STAGED_DASHBOARD=$(git diff --cached --name-only --diff-filter=ACM | grep -E '^pluribus_next/dashboard/src/.*\.(tsx?|jsx?)$' || true)

if [ -n "$STAGED_DASHBOARD" ]; then
    echo "=========================================="
    echo "  DASHBOARD GUARD: Pre-commit typecheck"
    echo "=========================================="
    echo "Checking staged files:"
    echo "$STAGED_DASHBOARD" | head -10
    echo ""

    cd "$DASHBOARD_DIR"

    if npm run typecheck > /tmp/typecheck-output.txt 2>&1; then
        echo "✓ Typecheck PASSED"
    else
        echo "✗ Typecheck FAILED"
        echo ""
        echo "Errors:"
        cat /tmp/typecheck-output.txt | grep -E "error TS|Error:" | head -20
        echo ""
        echo "=========================================="
        echo "  COMMIT BLOCKED - Fix TypeScript errors"
        echo "=========================================="

        emit_bus_event "git.commit.blocked" "metric" "{\"reason\":\"typecheck_failed\",\"actor\":\"${PLURIBUS_ACTOR:-unknown}\"}"

        exit 1
    fi

    cd "$REPO_ROOT"
fi

# --- Guard 4: Ring 0 protection ---
STAGED_RING0=$(git diff --cached --name-only | grep -E '^(AGENTS\.md|\.pluribus/constitution\.md|\.pluribus/luca\.json|pluribus_next/tools/iso_git\.mjs)$' || true)

if [ -n "$STAGED_RING0" ]; then
    echo ""
    echo "=========================================="
    echo "  ⚠️  RING 0 FILES MODIFIED"
    echo "=========================================="
    echo "The following protected files are staged:"
    echo "$STAGED_RING0"
    echo ""
    echo "Ring 0 modifications require HGT guard ladder."
    echo "Use iso_git.mjs for proper verification."
    echo ""

    emit_bus_event "git.ring0.warning" "metric" "{\"files\":\"$STAGED_RING0\",\"actor\":\"${PLURIBUS_ACTOR:-unknown}\"}"
fi

# --- Success ---
emit_bus_event "git.commit.approved" "metric" "{\"actor\":\"${PLURIBUS_ACTOR:-unknown}\",\"staged_count\":$(git diff --cached --name-only | wc -l)}"

exit 0
