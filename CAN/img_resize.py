import os
from PIL import Image

# List style
style_names = os.listdir('./wikiart/wikiart_original/')

# Loop to resize all styles

for i in range(len(style_names)):
    # create a folder
    if style_names[i] not in os.listdir('./wikiart/wikiart_16_9/'):
        os.mkdir('./wikiart/wikiart_16_9/' + style_names[i])
    # read images
    images = os.listdir('./wikiart/wikiart_original/' + style_names[i])
    
    for j in range(len(images)):
        my_img = Image.open('./wikiart/wikiart_original/' + style_names[i] + '/' + images[j])
        new_image = my_img.resize((256, 144))
        new_image.save('./wikiart/wikiart_16_9/' + style_names[i] + '/' + images[j])
        if j % 100 == 0:
            print(f'folder: {i+1}/{len(style_names)}, image:{j+1}/{len(images)}')
