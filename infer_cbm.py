import torch
import torch.nn as nn
from PIL import Image
import splice
import os

# Configuration
splice_model_name = "open_clip:ViT-B-32"
vocabulary = "laion"
vocabulary_size = 10000
l1_penalty = 0.25
device = 'cuda' if torch.cuda.is_available() else 'cpu'
predictor_path = 'cbm_predictor.pth'
num_classes = 5
label_to_class = {0: 'Chihuahua', 1: 'tiger cat', 2: 'goldfish', 3: 'robin', 4: 'brown bear'}

# Load Splice model
print("Loading Splice model...")
splice_model = splice.load(
    name=splice_model_name,
    vocabulary=vocabulary,
    vocabulary_size=vocabulary_size,
    device=device,
    l1_penalty=l1_penalty,
    return_weights=True
)
preprocess = splice.get_preprocess(splice_model_name)


# Predictor model
class CBMPredictor(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CBMPredictor, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


predictor = CBMPredictor(10000, num_classes).to(device)
predictor.load_state_dict(torch.load(predictor_path))
predictor.eval()

def intervene_concepts(activations, interventions=None):
    """Apply a dictionary of concept interventions (index -> new value) to activations."""
    if interventions is None:
        return activations

    out = activations.clone()
    for idx, value in interventions.items():
        if 0 <= idx < out.shape[0]:
            out[idx] = value
        else:
            raise IndexError(f"Concept index {idx} out of range [0, {out.shape[0]-1}]")
    return out


def infer(image_path, interventions=None, use_logic=False):
    # Load and preprocess image
    img = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Get concept activations
    with torch.no_grad():
        concept_activations = splice_model.encode_image(img).squeeze(0).cpu()

    print(f"Concept activations shape: {concept_activations.shape}")
    print(f"Non-zero concepts: {(concept_activations != 0).sum().item()}")
    print(f"Top 5 concepts: {torch.topk(concept_activations, 5)}")

    if interventions is not None:
        concept_activations = intervene_concepts(concept_activations, interventions)
        print(f"Applied interventions: {interventions}")
        print(f"Non-zero after intervention: {(concept_activations != 0).sum().item()}")
        print(f"Top 5 concepts after intervention: {torch.topk(concept_activations, 5)}")

    # Predict class
    with torch.no_grad():
        output = predictor(concept_activations.unsqueeze(0).to(device))
        probs = torch.softmax(output, dim=-1).squeeze(0).cpu()
        _, predicted = torch.max(probs, 0)
        predicted_class = label_to_class[predicted.item()]

    print(f"Predicted class: {predicted_class}")
    print(f"Probability vector: {probs.numpy()}")
    return concept_activations, predicted_class, probs


if __name__ == "__main__":
    # Example usage
    image_path = "000000308175.jpg" # "/datasets/imagenet/train/n02085620/n02085620_10011.JPEG"  # Example Chihuahua image
    print('--- Baseline ---')
    infer(image_path)

    print('\n--- Intervention: force concept 2760 to 0.0 ---')
    infer(image_path, interventions={2760: 0.0})

    print('\n--- Intervention: force concept 2760 to 1.0 ---')
    infer(image_path, interventions={2760: 1.0})