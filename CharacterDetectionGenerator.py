from CharacterGenerator import create_char_background, rotate, add_noise, blur, crop, perspective_transform
from CharacterGenerator import random_background_color, draw_text
from CharacterGenerator import add_outline, add_shadow, font_source
from CharacterSource import NumericCharacterSource, AlphaNumericCharacterSource
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageTransform, ImageChops
import random
import numpy as np
from Rectangle import Rectangle
from Point import Point
import Drawing

num_char_columns = 32
num_char_rows = 16
debug = True
char_source = NumericCharacterSource()


def random_offset(amp):
    return (random.random() - 0.5) * 2.0 * amp


def create_detection_example(image_width, image_height, options={}):
    font = font_source.random_font({'min_size':0.125, 'max_size':3.0})
    image_size = Point(image_width, image_height)
    canvas_size = 2 * image_size
    canvas_width = image_width * 2
    canvas_height = image_height * 2
    canvas_rect = Rectangle.from_point_and_size(Point(0,0), 2 * image_size)
    min_color_delta = options.get('min_color_delta', 32)
    text_color = random.randint(0,255)
    background_color = random_background_color(text_color, min_color_delta=min_color_delta)
    text = char_source.random_char()

    image = Drawing.create_noise_background(canvas_size, text_color, background_color, min_color_delta, random.uniform(0.5,1.5), max_factor=8)
    if random.random() < 0.3:
        image = crop(image)
        image = blur(image, options)
        image = add_noise(image, options)
        return image, 0
    else:
        label = True

    char_image = Image.new('RGBA', (canvas_width, canvas_height), (0,0,0,0))
    is_word_start = random.random() > 0.5
    is_word_end = random.random() > 0.5

    (w,h) = font.calc_text_size(text)
    x = 0.5 * (canvas_width - w)
    y = 0.5 * (canvas_height - h)

    #if float(h) / image_height < 0.25:
    #    label = False

    while not is_word_start and random.random() > 0.5:
        text = char_source.random_char() + text
        (w2,h2) = font.calc_text_size(text)
        x -= (w2 - w)
        w2 = w

    while not is_word_end and random.random() > 0.5:
        text = text + char_source.random_char()

    text_width, text_height = font.calc_text_size(text)
    text_size = Point(text_width, text_height)
#    x += random.randint(-2,2)
    x += random_offset(0.5 * image_width)
    y += random_offset(0.5 * image_height)
    char_rect = Rectangle.from_point_and_size(Point(x,y), text_size)
    image_rect = Rectangle.from_center_and_size(canvas_rect.center(), image_size)
    center_rect = image_rect.shrink_with_factor(Point(0.5, 0.5))
    label = image_rect.intersects_horizontally(char_rect) \
        and (center_rect.calc_vertical_overlap(char_rect) > 0.8) \
        and char_rect.height() / image_rect.height() < 1.25 \
        and char_rect.height() / image_rect.height() > 0.5

    draw = ImageDraw.Draw(char_image)

    if random.random() > 0.5:
        add_outline(draw, x, y, font, text, text_color)

    draw_text(draw, x, y, text, font, text_color)

    if random.random() > 0.5:
        shadow_image = add_shadow(char_image, x, y, font, text, text_color)
        image = Image.alpha_composite(image, shadow_image)

    #draw.rectangle(char_rect.as_array(), outline=(255,255,255))

    char_image = Image.alpha_composite(image, char_image)
    char_image = rotate(char_image, options)
#    char_image = perspective_transform(char_image)
    char_image = crop(char_image)
    char_image = blur(char_image, options)
    char_image = add_noise(char_image, options)
    return char_image, int(label)


def CharacterDetectionGenerator(batchsize, options={}):
#    mean = options.get('mean', None)
#    std = options.get('std', None)
    image_width = options.get('image_width', 32)
    image_height = 32
    full_alphabet = options.get('full_alphabet', False)
    char_source = AlphaNumericCharacterSource() if full_alphabet else NumericCharacterSource()

    while True:
        x = []
        y = []
        for i in range(0,batchsize):
            image, label = create_detection_example(image_width, image_height, options)
            image_data = np.array(image).astype('float32')/255.0
            x.append(image_data.reshape(image_height,image_width,1))
            y.append(label)

        yield np.array(x),y


if __name__ == "__main__":
    image_width = 32
    image_height = 32
    overview_image = Image.new("L", (image_width * num_char_columns, image_height * num_char_rows), 255)
    overview_draw = ImageDraw.Draw(overview_image)

    options={'min_color_delta':32.0, 'min_blur':0.5, 'max_blur':0.5, 'max_rotation':5.0, 'min_noise':4, 'max_noise':4, 'include_word_end_segmentation':True}
    options['full_alphabet'] = False
    options['add_background_lines'] = False

    full_alphabet = options.get('full_alphabet', False)
    char_source = AlphaNumericCharacterSource() if full_alphabet else NumericCharacterSource()

    num_positives = 0

    for j in range(0,num_char_rows):
        for i in range(0,num_char_columns):
            image, label = create_detection_example(image_width, image_height, options)
            if label == 1:
                num_positives += 1
            overview_image.paste(image, (image_width*i, image_height*j))

            if debug:
                overview_draw.text((i * image_width, j * image_height + 20), str(label))

    overview_image.save("overview.png")
    num_examples = num_char_rows * num_char_columns
    print("%d of %d positive examples (%f)" % (num_positives, num_examples, float(num_positives)/num_examples))
