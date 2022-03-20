from typing import List, Optional, Dict
import torch

categories = list(
    "A    R    N    D    C    Q    E    G    H    I    L    K    M    F    P    S    T    W    Y    "
    "V    X".replace(" ", "")
)
num_categories = len(categories)
categories = dict(zip(categories, range(num_categories)))


def help_encode(sequence: List[int], max_length: int):
    counter = 0
    out = list()
    for value in sequence:
        letter = [0 for _ in range(num_categories)]
        letter[value] = 1
        out.append(letter)
        counter += 1
    for _ in range(counter, max_length):
        out.append([0 for _ in range(num_categories)])
    return out


def one_hot_encode(sequences: List[str]):
    max_length = len(max(sequences, key=len))
    data = [[categories[aa] for aa in item] for item in sequences]
    # one hot encode
    one_hot_encoded = [help_encode(seq, max_length) for seq in data]
    return 1.0 * torch.tensor(one_hot_encoded)
