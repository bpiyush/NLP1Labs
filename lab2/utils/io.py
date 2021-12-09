"""Input output routines."""
import json


# this function reads in a textfile and fixes an issue with "\\"
def filereader(path):
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().replace("\\","")
            

def load_txt(path: str):
    """Loads data from text file."""
    with open(path, "r") as f:
        data = f.read()
    return data.split("\n")


def load_json(path: str) -> dict:
    """Helper to load json file"""
    with open(path, 'rb') as f:
        data = json.load(f)
    return data


def save_json(data: dict, path: str):
    """Helper to save `dict` as .json file."""
    with open(path, 'w') as f:
        json.dump(data, f)
