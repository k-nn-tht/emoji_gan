import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img

input_folder = "./resized_images"
output_folder = "augmented_images"
augmentations_per_image = 10
image_size = (64, 64)

os.makedirs(output_folder, exist_ok=True)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,     # Optional; skip for directional/flag emojis
    fill_mode='nearest'       # Avoid black borders
)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        path = os.path.join(input_folder, filename)
        img = Image.open(path).convert("RGB").resize(image_size)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        prefix = os.path.splitext(filename)[0]
        i = 0
        for batch in datagen.flow(x, batch_size=1):
            aug_img = array_to_img(batch[0])
            aug_img.save(os.path.join(output_folder, f"{prefix}_aug{i}.png"))
            i += 1
            if i >= augmentations_per_image:
                break

print(f"Augmented images saved to: {output_folder}")