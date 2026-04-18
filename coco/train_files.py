import json
import os
import torch
import open_clip
from PIL import Image
import numpy as np
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

def precompute_clip_features(json_path, image_dir, output_dir, model, preprocess, text_embeddings, device):
    with open(json_path, 'r') as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for image_id, sample in tqdm(data['images'].items(), desc=f"Processing {os.path.basename(json_path)}"):
        file_name = sample['file_name']
        image_path = os.path.join(image_dir, file_name)

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Encode image and normalize embeddings for cosine similarity
        with torch.no_grad():
            image_embedding = model.encode_image(image_input)
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        # Compute concept activations (cosine similarity)
        concept_activations = (image_embedding @ text_embeddings.T).squeeze(0).cpu().numpy()

        # Save as numpy array
        output_file = os.path.join(output_dir, f"{image_id}.npy")
        np.save(output_file, concept_activations)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai') # TODO: switch to 'laion2b_s34b_b79k' when available
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # Compute text embeddings for COCO classes and normalize them
    text_tokens = tokenizer(COCO_CLASS_NAMES).to(device)
    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    # Paths
    train_json = '/workspaces/SpLiCE/coco/cocologic_train_final.json'
    val_json = '/workspaces/SpLiCE/coco/cocologic_test_final.json'
    train_image_dir = '/datasets/coco/images'
    val_image_dir = '/datasets/coco/images'
    output_base_dir = '/workspaces/SpLiCE/coco/clip_features'

    # Precompute for train
    train_output_dir = os.path.join(output_base_dir, 'train')
    precompute_clip_features(train_json, train_image_dir, train_output_dir, model, preprocess, text_embeddings, device)

    # Precompute for val
    val_output_dir = os.path.join(output_base_dir, 'val')
    precompute_clip_features(val_json, val_image_dir, val_output_dir, model, preprocess, text_embeddings, device)

    print("Precomputation complete!")

if __name__ == '__main__':
    main()
    # python coco/train_cocologic.py --model_type linear --use_clip_features --epochs 50