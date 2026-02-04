#!/bin/bash
set -e

TOOL="/pluribus/pluribus_next/tools/domain_registry.py"
PYTHON="/pluribus/.pluribus/venv/bin/python3"

echo "Registering domains..."

# Helper function
add_domain() {
    DOMAIN=$1
    shift
    TAGS=""
    for t in "$@"; do
        TAGS="$TAGS --tag $t"
    done
    $PYTHON $TOOL add $DOMAIN $TAGS --source "operator_request" --context "2025-12-15_formal_mapping"
}

add_domain "aiwa.re" "hyperautomata" "neurosymbolic" "kroma"
add_domain "boddah.org" "refutation" "epistolary" "lsa"
add_domain "chadai.dev" "devstack" "ci" "sota"
add_domain "cinemacloud.co" "media" "whitelabel" "distro"
add_domain "cinemacloud.net" "media" "whitelabel" "distro"
add_domain "kareem.movie" "media" "film" "co-produced"
add_domain "kro.ma" "edge" "sky" "browser"
add_domain "mindlike.org" "research" "cognitive" "lab"
add_domain "mirrorstudios.co" "media" "agentic" "studio"
add_domain "nsphe.re" "research" "hypersphere" "theory"
add_domain "oooo.art" "media" "portfolio"
add_domain "oooo.video" "media" "ugc" "ar-vr"
add_domain "qredit.net" "financial" "ecash" "qr"
add_domain "qride.org" "service" "geoloc" "disruption"
add_domain "qride.app" "service" "geoloc" "disruption"
add_domain "selfagency.org" "service" "autonomous" "human-first"
add_domain "selfagency.app" "service" "autonomous" "human-first"
add_domain "superllm.dev" "research" "fox" "chromatic" "learning"

echo "All domains registered."
