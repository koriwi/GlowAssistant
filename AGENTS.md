# Collaboration preferences

- Keep responses short, information-dense, and easy to scan. No walls of text.
- The user wants close control over scope and decisions. Do only the work explicitly requested; do not start unsolicited coding, refactors, or features.
- For broad or ambiguous requests, briefly outline the proposed scope and clarify consequential choices before implementing. Resolve routine details within agreed scope without repeated approval requests.
- Prefer simple, readable solutions. Avoid premature optimization, speculative abstractions, and unnecessary dependencies.
- Add comments only when the reason or behavior would otherwise be unclear. Keep them short; fragments are fine. Do not narrate obvious code.

# Repository

- Local Python 3.11+ prototype; eventual goal is a Home Assistant conversation agent integration.
- Install dependencies: `python -m pip install -r requirements.txt` in a virtual environment.
- Inspect built-in HA MCP: `python prototype.py --inspect`.
- Classify a prompt: `python prototype.py "your request"`.
- Run offline checks: `python -m unittest -v`.
- Classifier choices use snapshot-local IDs mapped to context entries, not HA entity IDs.
- `.env` contains credentials. Never print its values or commit it.
- Scope and implementation checklist: `plan.md`.
