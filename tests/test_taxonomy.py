#!/usr/bin/env python3
"""Unit tests for the rule-based taxonomy (run: python -I tests/test_taxonomy.py)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utilities.taxonomy import classify_species  # noqa: E402

CASES = {
    "": ("unspecified", "unspecified"),
    "unidentified": ("unspecified", "unspecified"),
    "Homo sapiens": ("human_self", "Homo sapiens (self)"),
    "Human endogenous retrovirus W": ("human_self", "Homo sapiens (self)"),
    "Severe acute respiratory syndrome coronavirus 2": ("virus", "Coronaviridae"),
    "SARS-CoV1": ("virus", "Coronaviridae"),
    "Betacoronavirus gravedinis": ("virus", "Coronaviridae"),
    "Dengue virus": ("virus", "Flaviviridae-Orthoflavivirus"),
    "Zika virus": ("virus", "Flaviviridae-Orthoflavivirus"),
    "Orthohepacivirus hominis": ("virus", "Hepacivirus (HCV)"),
    "Orthohepadnavirus hominoidei": ("virus", "Hepadnaviridae (HBV)"),
    "Human betaherpesvirus 5": ("virus", "Herpesviridae"),
    "human gammaherpesvirus 4": ("virus", "Herpesviridae"),
    "Influenza A virus": ("virus", "Orthomyxoviridae"),
    "Lentivirus humimdef1": ("virus", "Retroviridae"),
    "Orthopoxvirus vaccinia": ("virus", "Poxviridae"),
    "Alphavirus chikungunya": ("virus", "Togaviridae"),
    "human respiratory syncytial virus": ("virus", "Pneumoviridae"),
    "Mycobacterium tuberculosis": ("bacterium", "Mycobacterium"),
    "Plasmodium falciparum": ("parasite", "Plasmodium"),
    "Trypanosoma cruzi": ("parasite", "Trypanosoma"),
    "Triticum aestivum": ("other_eukaryote", "Triticum aestivum"),
}


def main() -> int:
    for name, want in CASES.items():
        got = classify_species(name)
        assert got == want, (name, got, want)
    print("all taxonomy tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
