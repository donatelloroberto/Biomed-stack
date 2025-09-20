import re

def redact_procedural(text: str) -> (str, bool):
    """Simple procedural heuristic: detects lab verbs / procedural words and redacts.
    Returns (possibly_redacted_text, flagged_bool)
    This is a starter: replace with a learned classifier for production.
    """
    procedural_terms = ['incubate', 'mix', 'sterile', 'culture', 'pipette', 'centrifuge', 'protocol', 'plate', 'inoculate']
    lowered = text.lower()
    for t in procedural_terms:
        if t in lowered:
            # flag and redact procedural passages
            return (re.split(r"\n", text)[0] + "\n\n[OUTPUT REDACTED: contains procedural content; forwarded to reviewer]", True)
    return (text, False)
