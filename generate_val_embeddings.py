import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import splice

# Configuration
selected_wnids = ['n02099601', 'n02099712', 'n02106662', 'n02108915', 'n02088364', 'n02123045', 'n02123394', 'n02123597', 'n02124075']  # Chihuahua, tiger cat, goldfish, robin, brown bear
data_root = '/datasets/imagenet/val'
output_dir = '/workspaces/SpLiCE/embeddings/splice_val'
mapping_file = os.path.join(output_dir, 'mapping.json')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
num_workers = 0

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load Splice model
print("Loading Splice model...")
model = splice.load(
    name="open_clip:ViT-B-32",
    vocabulary="laion",
    vocabulary_size=10000,
    device=device,
    l1_penalty=0.25,
    return_weights=True,
)
preprocess = splice.get_preprocess("open_clip:ViT-B-32")

# Load full dataset
print("Loading ImageNet val dataset...")
dataset = ImageFolder(root=data_root, transform=preprocess)

# Filter to selected classes
selected_class_indices = [dataset.class_to_idx[wnid] for wnid in selected_wnids if wnid in dataset.class_to_idx]
filtered_samples = [s for s in dataset.samples if s[1] in selected_class_indices]
dataset.samples = filtered_samples
dataset.targets = [s[1] for s in filtered_samples]

print(f"Total images to process: {len(dataset)}")

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Initialize mapping
mapping = {}

# Process batches
for batch_idx, (images, _) in enumerate(dataloader):
    images = images.to(device)
    embeddings = model.encode_image(images)  # Sparse weights

    # Save each embedding
    start_idx = batch_idx * batch_size
    for i, emb in enumerate(embeddings):
        sample_idx = start_idx + i
        if sample_idx >= len(dataset):
            break
        img_path = dataset.samples[sample_idx][0]
        wnid = os.path.basename(os.path.dirname(img_path))
        img_id = os.path.basename(img_path).split('.')[0]
        emb_path = os.path.join(output_dir, wnid, f"{img_id}.pth")
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)
        torch.save(emb.cpu(), emb_path)
        mapping[img_path] = emb_path

    if batch_idx % 10 == 0:
        print(f"Processed batch {batch_idx}")

print(f"Finished processing {len(dataset)} images")

# Save mapping
with open(mapping_file, 'w') as f:
    json.dump(mapping, f, indent=2)

print(f"Embeddings saved to {output_dir}")
print(f"Mapping saved to {mapping_file}")
print("Done!")