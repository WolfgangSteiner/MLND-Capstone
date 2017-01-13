import random

class WordSource(object):
    def __init__(self):
        with open("words.txt") as f:
            self.words = [l.strip() for l in f.readlines()]

    def random_word(self):
        return random.choice(self.words)
