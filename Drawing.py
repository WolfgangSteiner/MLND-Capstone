from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import random
import math


def blur(img, radius):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def get_color(color, alpha=255):
    return (color,color,color,alpha)


def random_blur(img, min_blur, max_blur):
    return blur(img, random.uniform(min_blur, max_blur))


def scale_image(img, factor):
    return img.resize((int(img.size[0] * factor), int(img.size[1] * factor)), Image.LANCZOS)


def create_noise_background(size, text_color, background_color, min_color_delta, blur_radius, min_factor=1.0, max_factor=32.0):
    if background_color < text_color:
        min_color = 0
        max_color = text_color - min_color_delta
    else:
        min_color = text_color + min_color_delta
        max_color = 255

    noise_amp = min(abs(max_color - background_color), abs(background_color - min_color)) * random.random()

    wn = int(size.x / random.uniform(min_factor, min(max_factor, size.x)))
    hn = int(size.y / random.uniform(min_factor, min(max_factor, size.y)))
    noise = np.random.rand(hn,wn) * noise_amp + background_color
    noise = np.clip(noise, 0.0, 255.0)
    img = Image.fromarray(noise).convert('RGBA')
    img = blur(img, blur_radius)
    img = img.resize((size.x, size.y), Image.LANCZOS)
    return img


def random_contrasting_color(color, min_contrast=16):
    while True:
        contrast_color = random.randint(0,255)
        # find a contrast_color that has a minimum amount of contrast against color:
        if abs(color - contrast_color) >= min_contrast :
            return contrast_color


def add_outline(draw, x, y, text, font, outline_color):
    draw_text(draw, x-1, y-1, text, font, outline_color)
    draw_text(draw, x+1, y-1, text, font, outline_color)
    draw_text(draw, x-1, y+1, text, font, outline_color)
    draw_text(draw, x+1, y+1, text, font, outline_color)


def add_shadow(image, background_image, x, y, font, char, text_color):
    shadow_image = Image.new('RGBA', image.size, (0,0,0,0))
    shadow_layer = Image.new('RGBA', image.size, (0,0,0,255))
    shadow_image = Image.composite(shadow_layer, shadow_image, image.split()[-1])
    result = Image.new('RGBA', image.size, (0,0,0,0))
    result.paste(shadow_image, (random.randint(-3,3), random.randint(-3,3)))

    for n in (0, 10):
        result = result.filter(ImageFilter.BLUR)

    return Image.alpha_composite(background_image, result)


def draw_line(draw, p1, p2, color, width=1, alpha=255):
    draw.line((p1[0],p1[1],p2[0],p2[1]), fill=get_color(color, alpha=alpha), width=width)


def draw_text(draw, x, y, text, font, color, outline_color=None):
    if not outline_color is None:
        add_outline(draw, x, y, text, font, outline_color)

    draw.text((x,y - font.image_font.getoffset(text)[1]), text, font=font.image_font, fill=get_color(color))


def draw_text_with_random_outline(draw, x, y, text, font, color):
    outline_color = None if random.random() > 0.5 else random_contrasting_color(color, min_contrast=8)
    draw_text(draw, x, y, text, font, color, outline_color)


def draw_random_line(canvas_width, canvas_height, draw, text_color, background_color, min_color_delta, oversampling=4):
    p1 = np.random.random(2) * canvas_width / 2 * oversampling
    angle = random.random() * math.pi
    length = random.random() * canvas_width / 2 * oversampling
    width = random.randint(1, canvas_width / 2 * oversampling)
    color = random_background_color(text_color, min_color_delta=min_color_delta)
    alpha = random.randint(64,255)
    p2 = p1 + np.array([math.cos(angle), math.sin(angle)]) * length
    draw_line(draw, p1, p2, color, width, alpha=255)


def add_random_lines(canvas_width, canvas_height, draw, text_color, background_color, min_color_delta, oversampling=4):
    while random.random() < 0.95:
        draw_random_line(canvas_width, canvas_height, draw, text_color, background_color, min_color_delta, oversampling=oversampling)


def add_noise(image, options={}):
    min_noise = options.get('min_noise', 8)
    max_noise = options.get('max_noise', 8)
    w,h = image.size
    noise = (np.random.rand(h,w) - 0.5) * (min_noise + random.randint(0,max_noise - min_noise))
    im_array = np.array(image.convert('L')).astype(np.float32)
    im_array = np.clip(im_array + noise, 0.0, 255.0)
    return Image.fromarray(im_array).convert('L')


def create_char_background(width, height, text_color, background_color, min_color_delta, options={}):
    add_background_lines = options.get('add_background_lines', True)
    oversampling = options.get('oversampling', 2)

    if add_background_lines:
        image = Image.new('RGBA', (width * oversampling, height * oversampling), get_color(background_color))
        draw = ImageDraw.Draw(image, 'RGBA')
        add_random_lines(width, height, draw, text_color, min_color_delta, oversampling)
        image = image.resize((width, height), resample=Image.LANCZOS)
        image = random_blur(image, {'min_blur':0.5, 'max_blur':4.0})
    else:
        image = Image.new('RGBA', (width, height), get_color(background_color))

    return image


def perspective_transform(char_image):
    (w,h) = char_image.size
    bounding_box = np.array([0,0,0,h,w,h,w,0]) + (np.random.rand(8) - 0.5) * w / 4
    transformation = ImageTransform.QuadTransform(bounding_box)
    return char_image.transform((w, h), transformation, resample=Image.BICUBIC)


def random_rotate(char_image, options={}):
    max_rotation=options.get('max_rotation', 5.0)

    if max_rotation > 0:
        angle = np.random.normal(0.0, max_rotation)
        return char_image.rotate(angle, resample=Image.BICUBIC, expand = 0)
    else:
        return char_image


def random_blur(char_image, options={}):
    min_blur = options.get("min_blur", 1.0)
    max_blur = options.get('max_blur', 2.0)
    radius = random.uniform(min_blur, max_blur)
    return char_image.filter(ImageFilter.GaussianBlur(radius=radius))


def crop(char_image, width=None, rescale=True):
    (w,h) = char_image.size
    y1 = h / 4
    y2 = h * 3 / 4

    if width != None:
        x1 = w / 2 - width / 2
        x2 = w / 2 + width / 2
    else:
        x1 = w / 4
        x2 = w * 3 / 4

    img = char_image.crop((x1, y1, x2, y2))

    if width != None and rescale:
        img = img.resize((w/2,h/2), resample=Image.BICUBIC)

    return img


def random_background_color(text_color, min_color_delta=32):
    while True:
        background_color = random.randint(0,255)
        # find a text color that has a minimum amount of contrast against background_color:
        if abs(text_color - background_color) > min_color_delta:
            return background_color


def random_line_color(text_color, background_color, min_color_delta=32):
    while True:
        line_color = random.randint(0,255)
        # find a text color that has a minimum amount of contrast against background_color:
        if abs(text_color - background_color) > min_color_delta:
            return background_color
