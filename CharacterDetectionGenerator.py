from CharacterGenerator import random_font, create_char_background, rotate, add_noise, blur, crop, perspective_transform
from CharacterGenerator import random_background_color, random_char, calc_text_size, draw_text
from CharacterGenerator import add_outline, add_shadow
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageTransform, ImageChops
import random
import numpy as np

num_char_columns = 32
num_char_rows = 32
debug = True

def create_detection_example(image_width, image_height, options={}):
    font_tuple = random_font({'min_size':0.25, 'max_size':2.0})
    canvas_width = image_width * 2
    canvas_height = image_height * 2
    font = font_tuple[1]
    font_name = font_tuple[0]
    min_color_delta = options.get('min_color_delta', 32)
    text_color = random.randint(0,255)
    background_color = random_background_color(text_color, min_color_delta=min_color_delta)
    text = random_char()

    image = create_char_background(canvas_width, canvas_height, text_color, background_color, min_color_delta, options=options)
    if random.random() < 0.25:
        image = crop(image)
        image = blur(image, options)
        image = add_noise(image, options)
        return image, 0

    char_image = Image.new('RGBA', (canvas_width, canvas_height), (0,0,0,0))
    label = False
    is_word_start = random.random() > 0.5
    is_word_end = random.random() > 0.5

    (w,h) = calc_text_size(text, font_tuple)
    if random.random() < 0.5:
        x = 0.5 * canvas_width
    else:
        x = 0.5 * (canvas_width - w)

    if is_word_end and label == True:
        x = 0.5 * canvas_width - w

    y = 0.5 * (canvas_height - h)

    while not is_word_start and random.random() > 0.5:
        text = random_char() + text
        (w2,h2) = calc_text_size(text, font_tuple)
        x -= (w2 - w)

    while not is_word_end and random.random() > 0.5:
        text = text + random_char()

#    x += random.randint(-2,2)
    y += (random.random() - 0.5) * image_height
    y1 = y - 0.5 * image_height
    y2 = y + h - 0.5 * image_height
    y -= font.getoffset(text)[1]

    label = (y1 > 0) and (y1 < 0.25 * h) and (y2 > image_height * 0.75) and (y2 < image_height)

    draw = ImageDraw.Draw(char_image)

    if random.random() > 0.5:
        add_outline(draw, x, y, font, text, text_color)

    draw_text(draw, x, y, text, font, text_color)

    if random.random() > 0.5:
        shadow_image = add_shadow(char_image, x, y, font, text, text_color)
        image = Image.alpha_composite(image, shadow_image)

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
    while True:
        x = []
        y = []
        for i in range(0,batchsize):
            image, label = create_detection_example(image_width, image_height, options)
            image_data = np.array(image).astype('float32')
            x.append(image_data.reshape(image_height,image_width,1))
            y.append(label)

        yield np.array(x),y


if __name__ == "__main__":
    image_width = 32
    image_height = 32
    overview_image = Image.new("L", (image_width * num_char_columns, image_height * num_char_rows), 255)
    overview_draw = ImageDraw.Draw(overview_image)
    options={'min_color_delta':32.0, 'min_blur':0.5, 'max_blur':0.5, 'max_rotation':5.0, 'min_noise':4, 'max_noise':4, 'include_word_end_segmentation':True}
    for j in range(0,num_char_rows):
        for i in range(0,num_char_columns):
            image, label = create_detection_example(image_width, image_height, options)
            overview_image.paste(image, (image_width*i, image_height*j))

            if debug:
                overview_draw.text((i * image_width, j * image_height + 20), str(label))

    overview_image.save("overview.png")
