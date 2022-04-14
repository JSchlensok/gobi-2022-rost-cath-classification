# Stepwise creation of temporal holdout set from CATH latest release from 17.03.22
from Bio.PDB import *
import os
from os import listdir
import shutil


def get_pdb_or_cif_files():

    # "cath-b-newest-all" from CATH FTP server from the 17.03.2022 contains all domains from CATH database including
    # the newest additions
    # (ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/daily-release/newest/)
    file_cath_b_newest_all = open("data/cath-b-newest-all.txt", "r").read()
    array_cath_b_newest_all = []
    # Read all entries and safe them into an array
    for line in file_cath_b_newest_all.split("\n"):
        array_cath_b_newest_all.append(line)
    print(f"Retrieved {len(array_cath_b_newest_all)} domains from newest CATH release")
    # Format the saved entries into a dictionary
    dict_cath_b_newest_all = {}
    for entry in array_cath_b_newest_all:
        pdb_id = entry.split(" ")[0]
        cath_id = entry.split(" ")[2]
        domain_boundrary = entry.split(" ")[3]
        dict_cath_b_newest_all[pdb_id] = [cath_id, domain_boundrary]
    # "cath-domain-list" from CATH FTP server from the 17.03.2022 contains all domains from CATH database excluding
    # the newest additions
    # (ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/all-releases/v4_3_0/cath-classification-data/)
    cath_domain_list = open("data/cath-domain-list.txt", "r").read()
    array_cath_domains = []
    cath_domain_list = cath_domain_list.split(
        "#---------------------------------------------------------------------"
    )[1].split("\n")
    for line in cath_domain_list:
        pdb_id = line.split("     ")[0]
        array_cath_domains.append(pdb_id)
    print(f"Retrieved {len(array_cath_domains)} domains from CATH database previous release")
    # Remove all files from the dict_cath_b_newest_all dixtionary which are allready included within the CATH domain
    # list
    for entry in array_cath_domains:
        if dict_cath_b_newest_all.keys().__contains__(entry):
            dict_cath_b_newest_all.pop(entry)
    print(f"Reduced to {len(dict_cath_b_newest_all)} domains in the newest release")
    # Download matching PDB or CIF files for the extracted IDs
    counter = 0
    for id in dict_cath_b_newest_all.keys():
        counter += 1
        if counter == 21:
            break
        if id != "":
            pdbl = PDBList(pdb=f"PDB/{id[:4]}")
            pdbl.retrieve_pdb_file(id[:4])
    # Clean up the directory to only have the PDB/CIF file within the folder
    for pdbDir in listdir("PDB"):
        for dir1 in listdir(f"PDB\\{pdbDir}"):
            if dir1.__contains__(".cif"):
                break
            for dir2 in listdir(f"PDB\\{pdbDir}\\{dir1}"):
                if dir2.__contains__(".cif"):
                    shutil.move(f"PDB\\{pdbDir}\\{dir1}\\{dir2}", f"PDB\\{pdbDir}\\{dir2}")
    for pdbDir in listdir("PDB"):
        for dir1 in listdir(f"PDB\\{pdbDir}"):
            if dir1.__contains__(".cif"):
                continue
            for dir2 in listdir(f"PDB\\{pdbDir}\\{dir1}"):
                os.remove(f"PDB\\{pdbDir}\\{dir1}\\{dir2}")
            os.rmdir(f"PDB\\{pdbDir}\\{dir1}")
    # Create a formated file for the extracted newest entries in CATH
    # ID - label - boundary
    new_entry_file = open("data/cath-b-newest-entries.txt", "w")
    for key in dict_cath_b_newest_all.keys():
        new_entry_file.write(
            f"{key}    {dict_cath_b_newest_all.get(key)[0]}    {dict_cath_b_newest_all.get(key)[1]}\n"
        )
