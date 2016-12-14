from PIL import Image, ImageFilter
import numpy as np
import random


def blur(img, radius):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def random_blur(img, min_blur, max_blur):
    return blur(img, random.uniform(min_blur, max_blur))


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
