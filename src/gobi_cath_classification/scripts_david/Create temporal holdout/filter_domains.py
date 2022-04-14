from difflib import SequenceMatcher
import math


def remove_sequence_similarities_to_data_set(
    path_to_data_set, path_to_file_to_be_compared, outfile
):
    present_domains = open(path_to_data_set, "r")
    present_sequences = []
    saved = 0
    discarded = 0
    for domain in present_domains.read().split("\n"):
        if domain == "":
            continue
        if not domain.__contains__(">"):
            present_sequences.append(domain.strip())
    new_domains = open(path_to_file_to_be_compared, "r")
    saved_domains = []
    for domain in new_domains.read().split("\n"):
        if domain == "":
            continue
        sequence = domain.split("    ")[3].strip()
        cutoff = math.ceil((len(sequence) / 100) * 20)
        print("\n\n")
        print(
            f"Testing: {domain.split('    ')[1].strip()} - LENGTH: {len(sequence)} - CUTOFF: {cutoff}"
        )
        print("-----------------------------------------------------------------")
        save = True
        percent = 0
        size = 0
        for seq in present_sequences:
            match = SequenceMatcher(None, seq, sequence, autojunk=False).find_longest_match(
                0, len(seq), 0, len(sequence)
            )
            size_match = match.size
            if size_match >= cutoff:
                save = False
                size = size_match
                percent = int((size_match / len(sequence)) * 100)
                break
            else:
                if size_match >= size:
                    size = size_match
                    percent = int((size_match / len(sequence)) * 100)
        if save:
            saved_domains.append(domain)
            print(
                f"SAVED: {domain.split('    ')[1].strip()} - SIMILARITY: {percent}% - MATCH SIZE:{size}"
            )
            saved = saved + 1
            print(f"SAVED: {saved} - DISCARDED: {discarded}")
        else:
            print(
                f"DISCARDED: {domain.split('    ')[1].strip()} - SIMILARITY: {percent}% - MATCH SIZE:{size}"
            )
            discarded = discarded + 1
            print(f"SAVED: {saved} - DISCARDED: {discarded}")
    output = open(outfile, "w")
    for domain in saved_domains:
        output.write(f"{domain}\n")


def remove_duplicates():
    # Remove redundancies with training, validation and test sets
    file_temporal_holdout = open("data/temporal_holdout_set.txt", "r").read()
    file_train74k = open("data/train74k.fasta", "r").read()
    file_test300 = open("data/test300.fasta", "r").read()
    file_val200 = open("data/val200.fasta", "r").read()

    present_domains = []
    for line in file_train74k.split("\n"):
        if line == "":
            continue
        if line.__contains__(">"):
            present_domains.append(line.split(">")[1].strip().upper())
    for line in file_test300.split("\n"):
        if line == "":
            continue
        if line.__contains__(">"):
            present_domains.append(line.split(">")[1].strip().upper())
    for line in file_val200.split("\n"):
        if line == "":
            continue
        if line.__contains__(">"):
            present_domains.append(line.split(">")[1].strip().upper())

    new_domains = {}
    for domain in file_temporal_holdout.split("\n"):
        if domain == "":
            continue
        if not present_domains.__contains__(domain.split("    ")[0].upper()):
            new_domains[domain.split("    ")[0]] = domain
    all_domains_non_redundant_with_training = open(
        "data/temporal_holdout_set_no_duplicates.txt", "w"
    )
    for key in new_domains.keys():
        all_domains_non_redundant_with_training.write(f"{new_domains.get(key)}\n")


def remove_internal_duplicates(path_to_holdout_set):
    holdout_set = open(path_to_holdout_set, "r")
    IDs = []
    Sequences = []
    Labels = []
    out = []
    for line in holdout_set.read().split("\n"):
        if line == "":
            continue
        if not (
            IDs.__contains__(line.split("    ")[0].strip())
            or Labels.__contains__(line.split("    ")[1].strip())
            or Sequences.__contains__(line.split("    ")[3].strip())
        ):
            IDs.append(line.split("    ")[0].strip())
            Labels.append(line.split("    ")[1].strip())
            Sequences.append(line.split("    ")[3].strip())
            out.append(line)
    outfile = open(f"data/holdout{len(IDs)}.text", "w")
    for entry in out:
        outfile.write(f"{entry}\n")
