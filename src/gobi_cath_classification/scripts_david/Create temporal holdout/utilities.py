from Bio.PDB import *


def return_chains(pdb_file: str) -> dict:
    # Use MMCIFParser to retrieve chains from pdb/cif files
    chains = {}
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)
    for model in structure:
        for chain in model:
            chain_ID = str(chain).split("=")[1].split(">")[0]
            chain_str = ""
            for residue in chain:
                chain_str += translate_amino_acids(str(residue).split(" ")[1])
            chains[chain_ID] = chain_str
    return chains


def translate_amino_acids(AA: str) -> str:
    # Translate amino acid 3-digit code into amino acid 1-digit code
    if AA.upper() == "Ala".upper():
        return "A"
    if AA.upper() == "Cys".upper():
        return "C"
    if AA.upper() == "Asp".upper():
        return "D"
    if AA.upper() == "Glu".upper():
        return "E"
    if AA.upper() == "Phe".upper():
        return "F"
    if AA.upper() == "Gly".upper():
        return "G"
    if AA.upper() == "His".upper():
        return "H"
    if AA.upper() == "Ile".upper():
        return "I"
    if AA.upper() == "Lys".upper():
        return "K"
    if AA.upper() == "Leu".upper():
        return "L"
    if AA.upper() == "Met".upper():
        return "M"
    if AA.upper() == "Asn".upper():
        return "N"
    if AA.upper() == "Pro".upper():
        return "P"
    if AA.upper() == "Gln".upper():
        return "Q"
    if AA.upper() == "Arg".upper():
        return "R"
    if AA.upper() == "Ser".upper():
        return "S"
    if AA.upper() == "Thr".upper():
        return "T"
    if AA.upper() == "Val".upper():
        return "V"
    if AA.upper() == "Trp".upper():
        return "W"
    if AA.upper() == "Tyr".upper():
        return "Y"
    return ""
