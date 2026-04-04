import torch
import splice
import os

# Define the directories to process (WordNet IDs)
directories = ['n02099601', 'n02099712', 'n02106662', 'n02108915', 'n02088364', 'n02123045', 'n02123394', 'n02123597', 'n02124075']

# Create the output directory if it doesn't exist
output_folder = "concept_distribution"
os.makedirs(output_folder, exist_ok=True)

# Get the vocabulary once
print("Loading vocabulary...")
voc = splice.get_vocabulary('laion', 10000)

# Process each directory
for dir_name in directories:
    path = f'embeddings/splice/{dir_name}'
    if not os.path.exists(path):
        print(f"Directory {path} does not exist. Skipping.")
        continue
    
    files = [f for f in os.listdir(path) if f.endswith('.pth')]
    if not files:
        print(f"Directory {dir_name} is empty. Skipping.")
        continue
        
    print(f"Processing {len(files)} files for class {dir_name}...")
    
    # Create a single PyTorch tensor to keep a running tally of all 10,000 concepts
    concept_tallies = torch.zeros(10000, dtype=torch.int32)
    
    # Process each .pth file
    for file in files:
        file_path = os.path.join(path, file)
        w = torch.load(file_path).flatten()
        
        # Instantly add the non-zero locations to our master tally
        concept_tallies += (w != 0).int()
    
    # Find which concepts have a tally greater than 0
    active_indices = torch.nonzero(concept_tallies, as_tuple=True)[0]
    
    # Prepare the output text file
    output_file_path = os.path.join(output_folder, f"{dir_name}.txt")
    
    with open(output_file_path, "w") as f:
        if len(active_indices) > 0:
            # Match the indices to words
            results = [(voc[idx], concept_tallies[idx].item()) for idx in active_indices]
            
            # Sort by count, descending
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Write the results to the file
            for concept, count in results:
                f.write(f"{concept}: {count}\n")
            print(f" -> Saved {len(results)} concepts to {output_file_path}")
        else:
            f.write("No concepts found.\n")
            print(f" -> No concepts found. Saved empty state to {output_file_path}")

print("\nDone! All distributions have been written to the 'concept_distribution' folder.")