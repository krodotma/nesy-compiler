---- MODULE security ----
\* ARK LTL Spec: Security Invariants
\* This simplified TLA+ format is parsed by SpecLoader

FORMULA "□ (commit → ¬inject_code)"

INVARIANT NoDelete<auth_check>
INVARIANT Preserve<validate_input>
INVARIANT Preserve<sanitize>

FORBIDDEN "os\.system\("
FORBIDDEN "subprocess\.call\(.*shell=True"
FORBIDDEN "pickle\.loads\("
FORBIDDEN "__import__\("

ALLOWED "logging\."
ALLOWED "hashlib\."
ALLOWED "hmac\."

====
