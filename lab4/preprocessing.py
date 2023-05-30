import glob
import os
from PIL import Image
from tqdm import tqdm

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


for i in tqdm(glob.glob(os.path.join("data", "*.jpeg"))):
    new_path = i.replace('data', 'processed_data')
    img = Image.open(i)
    width, height = img.size
    new_width = min(width, height)

    left = (width - new_width)/2
    top = (height - new_width)/2
    right = (width + new_width)/2
    bottom = (height + new_width)/2

    # Crop the center of the image
    im = img.crop((left, top, right, bottom))
    im = im.resize((512,512))
    im = im.save(new_path)




