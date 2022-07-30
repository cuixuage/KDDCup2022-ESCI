import multiprocessing
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from multiprocesspandas import applyparallel
import string
import os
import sys
import re
from time import time
from collections import defaultdict
from gensim.models import KeyedVectors
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

wv = KeyedVectors.load("./output_5epoch/word2vec.wordvectors", mmap='r')
vector = wv['dog']  # Get numpy vector of a word
print(vector)
vector = wv['cat']  # Get numpy vector of a word
print(vector)

sim = wv.similarity('dog', 'cat')
sim = wv.similarity('car', 'cat')
print(sim)