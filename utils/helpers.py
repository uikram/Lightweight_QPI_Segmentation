import torch
import numpy as np
import random
from PIL import Image

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pad_to_square(img, background_color=(0, 0, 0)):
    w, h = img.size
    if w == h: return img
    max_size = max(w, h)
    new_img = Image.new('RGB', (max_size, max_size), background_color)
    new_img.paste(img, ((max_size - w) // 2, (max_size - h) // 2))
    return new_img