from PIL import Image, ImageFilter
import numpy as np
import random


def blur(img, radius):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def random_blur(img, min_blur, max_blur):
    return blur(img, random.uniform(min_blur, max_blur))


def create_noise_background(size, background_color, noise_amount, blur_radius):
    wn = int(size.x / random.uniform(1.0, 32.0))
    hn = int(size.y / random.uniform(1.0, 32.0))
    noise = (np.random.rand(hn,wn) - 0.5) * noise_amount + background_color
    noise = np.clip(noise, 0.0, 255.0)
    img = Image.fromarray(noise).convert('RGBA')
    img = blur(img, blur_radius)
    img = img.resize((size.x, size.y), Image.LANCZOS)
    return img
