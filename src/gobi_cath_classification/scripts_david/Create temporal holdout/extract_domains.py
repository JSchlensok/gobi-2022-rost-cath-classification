from typing import List
from utilities import return_chains


def extract_domains():
    new_entries = open("cath latest release data/cath-b-newest-entries.txt", "r")
    full_file = open("cath latest release data/temporal_holdout_set.txt", "w")
    count_total = 36572
    counter = 0
    for entry in new_entries.read().split("\n"):
        counter += 1
        try:
            pdb_id = entry.split("    ")[0][:4]
            cath_annotation = entry.split("    ")[1]
            domain_boundaries = entry.split("    ")[2]
            domain_sequence = ""
            len_sequence = 0
            for boundary in domain_boundaries.split(","):
                chain = boundary.split(":")[1]
                lower_boundary = int(boundary.split(":")[0].split("-")[0])
                upper_boundary = int(boundary.split(":")[0].split("-")[1])
                len_sequence += upper_boundary - lower_boundary + 1
                domain_sequence += retrieve_domains(pdb_id, chain, [lower_boundary, upper_boundary])
            if (
                not domain_sequence.__contains__("No sequence available for id")
                and not domain_sequence == ""
                and len(domain_sequence) == len_sequence
            ):
                print(f"Iteration {counter} --- {counter / count_total}% finished")
                print(
                    f"{entry.split('    ')[0]}    {cath_annotation}    {domain_boundaries}    {domain_sequence}"
                )
                full_file.write(
                    f"{entry.split('    ')[0]}    {cath_annotation}    {domain_boundaries}    {domain_sequence}\n"
                )
        except Exception as e:
            print(f"ERROR for domain {entry.split('    ')[0]}: {e}")


def retrieve_domains(file_name: str, chain: str, boundaries: List) -> str:
    try:
        dict_chains = return_chains(f"PDB\\{file_name}\\{file_name}.cif")
        chain = dict_chains.get(chain)
        lower_boundary = int(boundaries[0]) - 1
        if lower_boundary < 0:
            lower_boundary = 0
        upper_boundary = int(boundaries[1])
        return str(chain)[lower_boundary:upper_boundary]
    except Exception as e:
        return f"No sequence available for id {file_name} - {e}"
