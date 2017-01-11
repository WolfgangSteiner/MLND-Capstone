import pickle
import random
import os, glob
from PIL import ImageFont
from Font import Font
import re

class FontSource(object):
    font_blacklist = (
        # Linux fonts
        "FiraMono", "Corben", "D050000L","jsMath", "Redacted",
        "RedactedScript", "AdobeBlank", "EncodeSans", "cwTeX", "Droid", "Yinmar", "Lao",
        # Apple fonts
        "Apple Braille", "NISC18030", "Wingdings", "Webdings", "LastResort",
        "Bodoni Ornaments", "Hoefler Text Ornaments", "ZapfDingbats", "Kokonor",
        "Farisi", "Symbol", "Diwan Thuluth", "Diwan")


    def __init__(self):
        self.min_size = 0.75
        self.max_size = 1.0
        self.char_height = 32

        try:
            print("Loading fonts from font_cache.pickle ...")
            file = open('font_cache.pickle', 'rb')
            self.font_array = pickle.load(file)
            print "Loaded %d fonts" % len(self.font_array)
        except IOError:
            self.add_fonts()
            file = open('font_cache.pickle', 'wb')
            print ("Writing font_cache.pickle ...")
            pickle.dump(self.font_array, file, -1)


    # Blacklist symbol fonts and fonts not working with PIL
    @staticmethod
    def is_font_blacklisted(font_file):
        pattern = re.compile("^[A-Z]")
        font_family = os.path.basename(font_file).split(".")[0].split("-")[0]
        return font_family.startswith(FontSource.font_blacklist) or not pattern.match(font_family)


    def is_latin_font(self, font_subdir):
        try:
            fo = open(font_subdir + "/METADATA.pb", "r")
            return 'subsets: "latin"\n' in fo.readlines()
        except:
            return False


    def calc_font_size(self, font_file):
        font_size = 16
        text_height = 0
        font = None

        while text_height < self.char_height * 0.9:
            font = Font.from_ttf_file(font_file, font_size)
            _,text_height = font.calc_text_size("0123456789")
            font_size += 1

        return font_size - 1


    def add_fonts_in_subdir(self, directory_path):
        for font_file in glob.iglob(directory_path + "/*.ttf"):
            if not FontSource.is_font_blacklisted(font_file):
                try:
                    max_font_size = self.calc_font_size(font_file)
                    self.font_array.append((font_file, max_font_size))
                    print("adding font: %s, size %d" % (font_file, max_font_size))
                except IOError:
                    print("Error loading font: %s" % font_file)


    # Collect all ttf fonts in one font location, except those blacklisted.
    def add_fonts_in_directory(self, directory_path):
        for font_subdir in glob.iglob(directory_path + "/*"):
            if self.is_latin_font(font_subdir):
                self.add_fonts_in_subdir(font_subdir)
            else:
                print("Skipping non-latin fonts in : %s" % font_subdir)


    def add_fonts(self):
        self.font_array = []
        for font_dir in ("fonts-master/ofl", "fonts-master/apache", ):
            self.add_fonts_in_directory(font_dir)


    def random_font_with_size(self, size_factor):
        font_name, max_font_size = random.choice(self.font_array)
        size = int(max_font_size * size_factor)
        return Font.from_ttf_file(font_name, size)


    def random_font(self, options={}):
        min_size = options.get('min_size', 0.75)
        max_size = options.get('max_size', 1.0)
        size = random.uniform(min_size, max_size)
        return self.random_font_with_size(size)


    def random_font_from_normal_distribution(self, mean, sigma):
        size = random.gauss(mean, sigma)
        return self.random_font_with_size(size)
