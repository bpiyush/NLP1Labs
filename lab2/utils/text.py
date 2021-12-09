"""Text and text processing utilities."""
import re
from nltk import Tree
from nltk.treeprettyprinter import TreePrettyPrinter


SHIFT = 0
REDUCE = 1


def tokens_from_treestring(s):
    """extract the tokens from a sentiment tree"""
    return re.sub(r"\([0-9] |\)", "", s).split()


def transitions_from_treestring(s):
    """Extracts the SHIFT/REDUCE transitions from a sentiment tree"""
    s = re.sub("\([0-5] ([^)]+)\)", "0", s)
    s = re.sub("\)", " )", s)
    s = re.sub("\([0-4] ", "", s)
    s = re.sub("\([0-4] ", "", s)
    s = re.sub("\)", "1", s)
    return list(map(int, s.split()))