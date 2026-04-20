import numpy as np

import os

from tqdm import tqdm



# COCO class names

COCO_CLASS_NAMES = [

    'person',

    'bicycle',

    'car',

    'motorcycle',

    'airplane',

    'bus',

    'train',

    'truck',

    'boat',

    'traffic light',

    'fire hydrant',

    'stop sign',

    'parking meter',

    'bench',

    'bird',

    'cat',

    'dog',

    'horse',

    'sheep',

    'cow',

    'elephant',

    'bear',

    'zebra',

    'giraffe',

    'backpack',

    'umbrella',

    'handbag',

    'tie',

    'suitcase',

    'frisbee',

    'skis',

    'snowboard',

    'sports ball',

    'kite',

    'baseball bat',

    'baseball glove',

    'skateboard',

    'surfboard',

    'tennis racket',

    'bottle',

    'wine glass',

    'cup',

    'fork',

    'knife',

    'spoon',

    'bowl',

    'banana',

    'apple',

    'sandwich',

    'orange',

    'broccoli',

    'carrot',

    'hot dog',

    'pizza',

    'donut',

    'cake',

    'chair',

    'couch',

    'potted plant',

    'bed',

    'dining table',

    'toilet',

    'tv',

    'laptop',

    'mouse',

    'remote',

    'keyboard',

    'cell phone',

    'microwave',

    'oven',

    'toaster',

    'sink',

    'refrigerator',

    'book',

    'clock',

    'vase',

    'scissors',

    'teddy bear',

    'hair drier',

    'toothbrush',

]



def main():

    directory = "clip_features/train"

    if not os.path.exists(directory):

        print(f"Directory {directory} does not exist.")

        return

   

    files = [f for f in os.listdir(directory) if f.endswith('.npy')]

    if not files:

        print(f"No .npy files found in {directory}.")

        return

       

    print(f"Processing {len(files)} files...")

   

    # Initialize sum array

    sum_array = np.zeros(80, dtype=np.float32)

   

    # Process each file

    for file in tqdm(files, desc="Loading files"):

        file_path = os.path.join(directory, file)

        data = np.load(file_path)

        sum_array += data

   

    # Compute average

    average = sum_array / len(files)

   

    # Prepare results

    results = [(COCO_CLASS_NAMES[i], average[i]) for i in range(80)]

   

    # Sort by value descending

    results.sort(key=lambda x: x[1], reverse=True)

   

    # Write to txt file

    output_file = "verteilung.txt"

    with open(output_file, "w") as f:

        for concept, value in results:

            f.write(f"{concept}: {value}\n")

   

    print(f"Results written to {output_file}")



if __name__ == '__main__':

    main()