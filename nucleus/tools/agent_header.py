#!/usr/bin/env python3
"""
agent_header.py - Single-call header hydration for all Pluribus agents

Usage:
    python3 agent_header.py [agent_name]

    # Auto-detect from PLURIBUS_ACTOR env:
    PLURIBUS_ACTOR=gemini-cli python3 agent_header.py

    # Explicit:
    python3 agent_header.py claude
    python3 agent_header.py gemini
    python3 agent_header.py codex
    python3 agent_header.py qwen

Returns fully hydrated UNIFORM panel + minimal config.
No recursive file reads. No external dependencies.
"""

import os
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOL VERSIONS - SINGLE SOURCE OF TRUTH
# ═══════════════════════════════════════════════════════════════════════════════
UNIFORM_VERSION = "v1.4"
DKIN_VERSION = "v29"
PAIP_VERSION = "v15"
CITIZEN_VERSION = "v1"


def detect_agent() -> str:
    """Auto-detect agent from environment."""
    actor = os.environ.get("PLURIBUS_ACTOR", "")
    if actor:
        # Normalize: gemini-cli -> gemini, claude_opus -> claude
        return actor.lower().replace("-cli", "").replace("_cli", "").split("_")[0].split("-")[0]
    return "agent"


def generate_panel(agent: str) -> str:
    """Generate the UNIFORM v1.4 panel for an agent."""
    # Pad agent name to 6 chars for alignment
    agent_padded = f"{agent:<6}"

    return f"""┌──────────────────────────────────────────────────────────────────────────────┐
│ UNIFORM {UNIFORM_VERSION} │ {agent_padded} │ sess:new │ cell:dia.1.0 │ lane:dialogos │ d:0      │
├──────────────────────────────────────────────────────────────────────────────┤
│ PROTO │ DKIN {DKIN_VERSION} │ PAIP {PAIP_VERSION} │ CITIZEN {CITIZEN_VERSION} │ gen:? │ lineage:?                │
│ SCORE │ ████████████ 100/100 ✓ │ CDI:? │ health:nominal                     │
│ STATE │ idle │ hexis:? │ bus:+0 │ uncommit:? │ branch:?                     │
│ SCOPE │ goal:"awaiting task" │ artifact:none                                │
└──────────────────────────────────────────────────────────────────────────────┘"""


def generate_config(agent: str) -> str:
    """Generate minimal config. Keep it SHORT to avoid loops."""
    return f"""
[Panel generated for {agent}. Now respond to user. Do not run this command again.]
"""


def get_full_header(agent: str = None) -> str:
    """Return complete hydrated header for an agent."""
    if not agent:
        agent = detect_agent()

    agent = agent.lower().strip()

    panel = generate_panel(agent)
    config = generate_config(agent)

    return panel + "\n" + config


def get_preamble(agent: str = None) -> str:
    """
    Return a preamble to inject into the FIRST prompt for agents that don't
    read config files as instructions (e.g., Qwen).

    Usage by other tools:
        preamble = subprocess.check_output(['python3', 'agent_header.py', '--preamble', 'qwen'])
        full_prompt = preamble.decode() + "\n\nUser's actual message: " + user_msg
    """
    if not agent:
        agent = detect_agent()

    agent = agent.lower().strip()
    panel = generate_panel(agent)

    return f"""[SYSTEM INSTRUCTION]
You are operating in the Pluribus multiagent system.
Start EVERY response with this exact panel:

{panel}

Then respond to the user. This is mandatory protocol.
[END SYSTEM INSTRUCTION]

"""


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate agent headers for Pluribus')
    parser.add_argument('agent', nargs='?', help='Agent name (claude, gemini, qwen, codex, grok)')
    parser.add_argument('--preamble', action='store_true',
                        help='Output preamble for prompt injection (for agents like Qwen)')
    args = parser.parse_args()

    agent = args.agent if args.agent else detect_agent()

    if args.preamble:
        print(get_preamble(agent))
    else:
        print(get_full_header(agent))


if __name__ == "__main__":
    main()
