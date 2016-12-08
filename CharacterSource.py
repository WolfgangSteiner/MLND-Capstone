import string
import random

class CharacterSource(object):
    def __init__(self):
        self.chars = ""

    def random_char(self):
        return random.choice(self.chars)

    def index_for_char(self, char):
        return self.chars.index(char)

    def num_chars(self):
        return len(self.chars)


class NumericCharacterSource(CharacterSource):
    def __init__(self):
        self.chars = '0123456789'


class AlphaNumericCharacterSource(CharacterSource):
    def __init__(self):
        self.chars = '0123456789' + string.ascii_lowercase  + string.ascii_uppercase
