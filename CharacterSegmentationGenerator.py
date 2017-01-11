from CharacterGenerator import font_source
from CharacterSource import NumericCharacterSource, AlphaNumericCharacterSource
from PIL import Image, ImageDraw
import random
import numpy as np
from FontSource import FontSource
import Drawing

num_char_columns = 32
num_char_rows = 32
char_source = NumericCharacterSource()

def create_segmentation_example(image_width, image_height, font, options={}):
    canvas_width = image_width * 2
    canvas_height = image_height * 2
    include_word_end_segmentation = options.get('include_word_end_segmentation', False)
    min_color_delta = options.get('min_color_delta', 32)
    text_color = random.randint(0,255)
    background_color = Drawing.random_background_color(text_color, min_color_delta=min_color_delta)
    text = char_source.random_char()

    image = Drawing.create_char_background(canvas_width, canvas_height, text_color, background_color, min_color_delta, options=options)
    if random.random() < 0.25:
        image = Drawing.crop(image)
        image = Drawing.random_blur(image, options)
        image = Drawing.add_noise(image, options)
        return image, 0

    char_image = Image.new('RGBA', (canvas_width, canvas_height), (0,0,0,0))
    label = False
    is_word_start = random.random() > 0.5
    is_word_end = random.random() > 0.5

    (w,h) = font.calc_text_size(text)
    if random.random() < 0.5:
        label = True
        x = 0.5 * canvas_width
    else:
        x = 0.5 * (canvas_width - w)
        label = False

    if is_word_end and label == True:
        x = 0.5 * canvas_width - w
        label = include_word_end_segmentation

    y = 0.5 * (canvas_height - h)

    if not is_word_start:
        text = char_source.random_char() + text
        (w2,h2) = font.calc_text_size(text)
        x -= (w2 - w)

    if not is_word_end:
        text = text + char_source.random_char()

#    x += random.randint(-2,2)
    y += (random.random() - 0.5) * (image_height - h)

    draw = ImageDraw.Draw(char_image)
    Drawing.draw_text_with_random_outline(draw, x, y, text, font, text_color)

    if random.random() > 0.5:
        image = Drawing.add_shadow(char_image, image, x, y, font, text, text_color)

    char_image = Image.alpha_composite(image, char_image)
    char_image = Drawing.random_rotate(char_image, options)
#    char_image = perspective_transform(char_image)
    char_image = Drawing.crop(char_image)
    char_image = Drawing.random_blur(char_image, options)
    char_image = Drawing.add_noise(char_image, options)
    return char_image, int(label)


def CharacterSegmentationGenerator(batchsize, options={}):
    full_alphabet = options.get('full_alphabet', False)
    if full_alphabet:
        char_source = AlphaNumericCharacterSource()

    image_width = options.get('image_width', 16)
    image_height = 32
    while True:
        x = []
        y = []
        for i in range(0,batchsize):
            font = font_source.random_font(options)
            is_char_border = int(random.random() > 0.5)
            image, label = create_segmentation_example(image_width, image_height, font, options)
            image_data = np.array(image).astype('float32')/255.0

            x.append(image_data.reshape(image_height,image_width,1))
            y.append(label)

        yield np.array(x),y


if __name__ == "__main__":
    image_width = 16
    image_height = 32
    overview_image = Image.new("L", (image_width * num_char_columns, image_height * num_char_rows), 255)
    overview_draw = ImageDraw.Draw(overview_image)
    options={'min_color_delta':32.0, 'min_blur':0.5, 'max_blur':0.5, 'max_rotation':5.0, 'min_noise':4, 'max_noise':4, 'include_word_end_segmentation':True}

    full_alphabet = options.get('full_alphabet', False)
    if full_alphabet:
        char_source = AlphaNumericCharacterSource()

    for j in range(0,num_char_rows):
        for i in range(0,num_char_columns):
            font=font_source.random_font(options)
            image, label = create_segmentation_example(image_width, image_height, font, options)
            overview_image.paste(image, (image_width*i, image_height*j))
            overview_draw.text((i * image_width, j * image_height + 20), str(label))

    overview_image.save("overview.png")
