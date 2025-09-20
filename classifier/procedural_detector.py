"""
Starter procedural detector.
This module provides a small heuristic-based detector for procedural/wet-lab content.
Replace with an ML classifier trained on your red-team dataset for production.
"""
import re

procedural_keywords = [
    'incubate','mix','sterile','culture','pipette','centrifuge','protocol','plate','inoculate',
    'incubation','add mg','hours at','wash with','prepare buffer','heat at'
]

def is_procedural(text: str) -> bool:
    t = text.lower()
    for k in procedural_keywords:
        if k in t:
            return True
    # simple regex for temperatures/volumes
    if re.search(r"\b\d+\s?(ml|µl|uL|g|mg|°C|C)\b", t):
        return True
    return False
