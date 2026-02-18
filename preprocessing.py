import re
import numpy as np
from collections import Counter

class TextProcessor:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.vocab_size = 0
        self.data = []

    def process(self, text, min_count=1):
        # basic cleanup
        text = re.sub(r'[^a-z\s]', '', text.lower())
        tokens = text.split()

        # filter rare words
        counts = Counter(tokens)
        tokens = [t for t in tokens if counts[t] >= min_count]
        
        vocab = sorted(list(set(tokens)))
        self.vocab_size = len(vocab)
        
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = {i: w for i, w in enumerate(vocab)}
        
        # pre-convert to IDs
        self.data = [self.word2id[t] for t in tokens]
        
        return self.data
