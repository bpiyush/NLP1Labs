"""Defines a class for an example sentence."""
from collections import namedtuple
from nltk import Tree

from utils.io import filereader
from utils.text import tokens_from_treestring, transitions_from_treestring


# A simple way to define a class is using namedtuple.
# This is the class that will contain all information
# about a single data point.
Example = namedtuple("Example", ["tokens", "tree", "label", "transitions"])

   
def examplereader(path, lower=False):
    """Returns all examples in a file one by one."""
    for line in filereader(path):
        line = line.lower() if lower else line
        tokens = tokens_from_treestring(line)
        tree = Tree.fromstring(line)  # use NLTK's Tree
        label = int(line[1])
        trans = transitions_from_treestring(line)
        yield Example(tokens=tokens, tree=tree, label=label, transitions=trans)
