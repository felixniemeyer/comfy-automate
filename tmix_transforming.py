"""Python script that temporaly interpolates frames according to weights and respecting known transformations."""
import os
import sys

import numpy as np

from PIL import Image

weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

comfy_path = os.path.expanduser('~/work/ai/ComfyUI')
folder = "frame_by_frame18"
folder_path = os.path.join(comfy_path, 'output', folder)

out_folder = os.path.join(comfy_path, 'output', folder + "_interpolated")

if not os.path.exists(folder_path):
    print("Folder does not exist:", folder_path)
    sys.exit(1)

if not os.path.exists(out_folder):
    os.makedirs(out_folder)
else:
    print("Folder already exists:", out_folder)
    empty = input("Do you want to empty it? (y/n)")
    if empty == "y":
        for file in os.listdir(out_folder):
            os.remove(os.path.join(out_folder, file))
    else:
        sys.exit(1)

print("Interpolating frames from")
print(folder_path)
print("to")
print(out_folder)

# go through all files like a kernel
frames = []
counter = 0
for file in sorted(os.listdir(folder_path)):
    if file.endswith(".png"):
        print(f"File: {file}")
        min_width = 2048
        min_height = 2048
        # rotate and scale all frames 
        for frame in frames:
            frame = frame.rotate(1)
            crop_border_px = 0.02 * frame.width
            frame = frame.crop((
                crop_border_px, 
                crop_border_px, 
                image.width - crop_border_px, 
                image.height - crop_border_px
            ))
            min_width = min(min_width, frame.width)
            min_height = min(min_height, frame.height)

        # load next image
        image = Image.open(os.path.join(folder_path, file)) 
        frames.append(image)

        if len(frames) == len(weights):
            print(f"Interpolating frame {counter}")
            print(f"min_width: {min_width}, min_height: {min_height}")

            # create numpy array for new image
            interpolated_frame = np.zeros((min_width, min_height, 3), dtype=np.float32)

            # add all frames weighted and centered
            for i, frame in enumerate(frames):
                print(f"Adding frame {i}/{len(frames)}")
                left = (frame.width - min_width) // 2
                top = (frame.height - min_height) // 2
                np_frame = np.array(frame)
                interpolated_frame += weights[i] * np_frame[left:left+min_width, top:top+min_height]

            print(f"Saving interpolated frame {counter}")

            # to uint8
            output_image = Image.fromarray(interpolated_frame.astype(np.uint8))
            output_image.save(os.path.join(out_folder, f"interpolated_{counter:09d}.png"))

            frames.pop(0)
            counter += 1

