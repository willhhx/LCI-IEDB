"""Rule-based source-organism categories and taxon groups (offline; no NCBI lineage available).

IEDB provides a species name and NCBITaxon id per peptide but no lineage.  We therefore use explicit, auditable rules:

* ``category``: virus | bacterium | parasite | fungus | human_self | other_eukaryote | unspecified
* ``group``: virus FAMILY-level (or sub-family where biology demands it, e.g. flavivirus vs hepacivirus), bacterial /
  parasite / fungal GENUS, otherwise the species name.  Groups are used for (i) the pathogen-held-out split S3 and
  (ii) linking homologous antigens (same group + same normalised protein name) for the antigen-cluster split S1.

These are heuristics; ``tests/test_taxonomy.py`` documents the expected assignments and
``data/metadata/taxon_groups.csv`` lists every assignment used in the release so users can audit or override it.
"""
from __future__ import annotations

import re

# (regex on lower-cased species name) -> group; first match wins
VIRUS_GROUPS: list[tuple[str, str]] = [
    (r"sars|coronavirus", "Coronaviridae"),
    (r"dengue|zika|west nile|yellow fever|japanese encephalitis|tick-borne|flavivirus", "Flaviviridae-Orthoflavivirus"),
    (r"hepacivirus|hepatitis c", "Hepacivirus (HCV)"),
    (r"hepadnavirus|hepatitis b", "Hepadnaviridae (HBV)"),
    (r"hepatitis delta", "Deltavirus (HDV)"),
    (r"hepevirus|hepatitis e", "Hepeviridae (HEV)"),
    (r"herpes|rhadinovirus|roseolovirus|cytomegalovirus|epstein", "Herpesviridae"),
    (r"influenza", "Orthomyxoviridae"),
    (r"lentivirus|retrovirus|hiv|htlv|deltaretro", "Retroviridae"),
    (r"poxvirus|vaccinia|variola", "Poxviridae"),
    (r"alphavirus|chikungunya", "Togaviridae"),
    (r"papillomavirus", "Papillomaviridae"),
    (r"polyomavirus", "Polyomaviridae"),
    (r"adenovirus", "Adenoviridae"),
    (r"respiratory syncytial|metapneumovirus|pneumovirus", "Pneumoviridae"),
    (r"mumps|measles|morbillivirus|rubulavirus|parainfluenza", "Paramyxoviridae"),
    (r"arenavirus|lassa|choriomeningitis", "Arenaviridae"),
    (r"ebola|marburg|filovirus", "Filoviridae"),
    (r"hantavirus", "Hantaviridae"),
    (r"norovirus|sapovirus", "Caliciviridae"),
    (r"rotavirus", "Reoviridae"),
    (r"parvovirus", "Parvoviridae"),
    (r"enterovirus|rhinovirus|hepatovirus|coxsackie|poliovirus|hepatitis a", "Picornaviridae"),
    (r"rift valley|phlebovirus", "Phenuiviridae"),
    (r"rabies|lyssavirus", "Rhabdoviridae"),
]
BACTERIA = ("mycobacter", "streptococc", "staphylococc", "francisella", "helicobacter", "chlamydia", "bacteroid", "flavonifractor",
            "dorea", "alistipes", "shigella", "klebsiella", "dickeya", "salmonella", "escherichia", "listeria", "neisseria",
            "bordetella", "burkholderia", "yersinia", "clostridi", "bacillus", "vibrio", "borrelia", "brucella", "legionella",
            "pseudomonas", "treponema", "mycoplasma", "haemophilus", "coxiella", "rickettsia", "campylobacter", "enterococc",
            "corynebacter", "lactobacill", "bifidobacter", "akkermansia", "faecalibacter", "prevotella", "ruminococc",
            "borreliella", "porphyromonas", "fusobacterium", "orientia", "pasteurella", "aeromonas", "aggregatibacter",
            "mannheimia", "prescottella", "leptospira", "eikenella", "shewanella", "serratia", "mycoplasmoides", "schaalia",
            "brachybacterium", "collimonas", "trichodesmium", "chelativorans")
PARASITES = ("plasmodium", "trypanosoma", "toxoplasma", "leishmania", "schistosoma", "brugia", "onchocerca", "cryptosporid",
             "entamoeba", "giardia", "babesia", "theileria", "echinococc", "taenia", "strongyloid", "ascaris", "necator",
             "fasciola", "trichomonas", "encephalitozoon")
FUNGI = ("candida", "aspergillus", "cryptococc", "histoplasma", "coccidioid", "pneumocystis", "saccharomyces",
         "purpureocillium", "alternaria", "penicillium")
OTHER_EUK = ("mus musculus", "triticum", "ambrosia", "malus", "spinacia", "dermatophagoides", "bos taurus", "arachis",
             "betula", "phleum", "lolium", "felis", "canis", "rattus", "sarcophilus", "gallus", "hevea", "glycine", "chironomus",
             "oryctolagus", "pan troglodytes", "cynodon", "viscum", "parietaria", "monodelphis", "poa ", "dendroaspis", "secale",
             "tetronarce", "callithrix")
PATHOGEN_CATEGORIES = frozenset({"virus", "bacterium", "parasite", "fungus"})


def classify_species(species: str) -> tuple[str, str]:
    """Return ``(category, group)`` for an IEDB species string."""
    s = (species or "").strip()
    low = s.lower()
    if low in ("", "unidentified", "unknown"):
        return "unspecified", "unspecified"
    if low == "homo sapiens" or "endogenous retrovirus" in low:
        return "human_self", "Homo sapiens (self)"
    if any(low.startswith(x) for x in OTHER_EUK):
        return "other_eukaryote", s
    if "virus" in low or "sars" in low:
        for pat, grp in VIRUS_GROUPS:
            if re.search(pat, low):
                return "virus", grp
        return "virus", s
    genus = s.split()[0]
    if any(low.startswith(x) for x in BACTERIA):
        return "bacterium", genus
    if any(low.startswith(x) for x in PARASITES):
        return "parasite", genus
    if any(low.startswith(x) for x in FUNGI):
        return "fungus", genus
    return "other", s
