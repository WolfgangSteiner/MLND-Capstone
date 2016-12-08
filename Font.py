from PIL import ImageFont

class Font(object):
    def __init__(self, font_name, image_font):
        self.font_name = font_name
        self.image_font = image_font


    def calc_text_size(self, text):
        try:
            (w,h) = self.image_font.getsize(text)
            h -= self.image_font.getoffset(text)[1]
            return (w,h)
        except IOError:
            print("font.getsize failed for font:%s" % self.font_name)
            raise IOError


    def getoffset(self, text):
        return self.image_font.getoffset(text)


    @staticmethod
    def from_ttf_file(ttf_file_name, font_size):
        image_font = ImageFont.truetype(font=ttf_file_name, size=font_size)
        return Font(ttf_file_name, image_font)
