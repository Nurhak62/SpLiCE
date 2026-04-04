import torch
import splice
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
directories = ['n02085620', 'n02123159', 'n01443537', 'n01534433', 'n02132136']
base_path = 'embeddings/splice'
output_folder = 'concept_plots'
plot_topk = 20 # Number of concepts to show in the bar plot

# Create the output directory
os.makedirs(output_folder, exist_ok=True)

# Load the vocabulary
print("Loading vocabulary...")
vocab = splice.get_vocabulary('laion', 10000)

for dir_name in directories:
    path = os.path.join(base_path, dir_name)
    if not os.path.exists(path):
        print(f"Directory {path} does not exist. Skipping.")
        continue
    
    files = [f for f in os.listdir(path) if f.endswith('.pth')]
    if not files:
        print(f"Directory {dir_name} is empty. Skipping.")
        continue
        
    print(f"Aggregating weights for {dir_name} ({len(files)} images)...")
    
    # Create a tensor to hold the sum of all weights
    sum_weights = torch.zeros(10000, dtype=torch.float32)
    
    for file in files:
        file_path = os.path.join(path, file)
        w = torch.load(file_path).flatten().cpu()
        sum_weights += w
        
    # Calculate the mean weight across all images in this class
    mean_weights = sum_weights / len(files)
    
    # Sort the mean weights in descending order
    sorted_weights, sorted_indices = torch.sort(mean_weights, descending=True)
    
    concept_names = []
    concept_weights = []
    
    # Grab the top K concepts (ignoring 0 weights)
    # --- THIS IS THE FIXED LOOP ---
    for i in range(plot_topk):
        weight = sorted_weights[i].item()
        if weight == 0:
            break # Stop if we run out of non-zero concepts
            
        vocab_idx = sorted_indices[i].item()
        concept_names.append(vocab[vocab_idx])
        concept_weights.append(weight)

    if not concept_names:
        print(f"No non-zero concepts found for {dir_name}.")
        continue

    # ==========================================
    # SpLiCE Repo Plotting Logic
    # ==========================================
    df = pd.DataFrame({"concept": concept_names, "weight": concept_weights})
    
    # Set style
    sns.set_style("darkgrid", {"axes.facecolor": "whitesmoke"})
    colors = ["#e86276ff", "#629d1eff"]
    custom_palette = sns.color_palette(colors)
    sns.set_palette(custom_palette, 2)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(y="concept", x="weight", data=df, orient='h')

    # Formatting
    title = f"Class {dir_name} Decomposition"
    plt.title(title, fontsize=20)
    plt.xlabel('Average Weight', fontsize=16)
    plt.ylabel('Concept', fontsize=16)
    
    if ax.get_legend() is not None:
        ax.get_legend().remove()
        
    sns.despine(bottom=True)
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(output_folder, f"{dir_name}_decomposition.pdf")
    plt.savefig(save_path)
    plt.close() # Clear the figure memory for the next loop
    
    print(f" -> Saved plot to {save_path}")

print("\nDone! All plots generated.")