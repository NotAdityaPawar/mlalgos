# Byte pair encoding 


from collections import Counter, defaultdict

class BPETokenizer:
    def __init__(self, vocab_size = 100):
        self.vocab_size = vocab_size
        self.merge = {}
        
        
    def train(self, text):
        words = text.split()
        
        vocab = Counter(words)