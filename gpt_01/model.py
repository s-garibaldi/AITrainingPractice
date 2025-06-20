import random
from collections import defaultdict

class NGramModel:
    """A simple n-gram language model for text generation."""
    def __init__(self, n=2):
        self.n = n
        self.ngrams = defaultdict(list)
        self.starts = []

    def train(self, text):
        """Train the model on a single line of text."""
        tokens = text.split()
        if len(tokens) < self.n:
            return
        for i in range(len(tokens) - self.n):
            key = tuple(tokens[i:i+self.n-1])
            self.ngrams[key].append(tokens[i+self.n-1])
        self.starts.append(tuple(tokens[:self.n-1]))

    def generate(self, max_len=20):
        """Generate text up to max_len tokens."""
        if not self.starts:
            return ""
        current = random.choice(self.starts)
        output = list(current)
        for _ in range(max_len - self.n + 1):
            next_words = self.ngrams.get(current)
            if not next_words:
                break
            next_word = random.choice(next_words)
            output.append(next_word)
            current = tuple(output[-(self.n-1):])
        return ' '.join(output) 