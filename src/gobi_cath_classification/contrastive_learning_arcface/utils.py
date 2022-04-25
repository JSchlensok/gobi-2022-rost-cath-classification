import os
from pathlib import Path
import socket


def get_base_dir() -> Path:
    if "ALLUSERSPROFILE" in os.environ:
        # Windows PC
        return Path(
            "G:\\My Drive\\Files\\Projects\\University\\2021W\\GoBi\\Project\\gobi-2022-rost-cath-classification"
        )
    elif socket.gethostname() == "thiccpad":
        return Path("/projects/University/2021W/GoBi/Project/gobi-2022-rost-cath-classification")
    else:
        return Path(
            "/content/drive/MyDrive/Files/Projects/University/2021W/GoBi/Project/gobi-2022-rost-cath-classification"
        )
