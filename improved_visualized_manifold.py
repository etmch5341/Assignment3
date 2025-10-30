# visualize_manifold.py
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os

from AE import MotionAutoencoder
from dataloader import CMUMotionDataset

def generate_tsne_visualization(model_path, data_dir, output_file, sample_size=1000):
    """
    Generates a t-SNE visualization of the learned motion manifold, colored by motion type.
    """
    print("🚀 Starting manifold visualization...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load the dataset to get samples and metadata
    print("Loading dataset...")
    dataset = CMUMotionDataset(
        data_dir=data_dir,
        frame_rate=30,
        window_size=160,
        overlap=0.5
    )

    # 2. Load the trained model
    print(f"Loading trained model from {model_path}...")
    sample = dataset[0]
    positions_flat = sample["positions_flat"]
    trans_vel_xz = sample["trans_vel_xz"]
    input_dim = positions_flat.shape[1] + trans_vel_xz.shape[1] + 1
    
    model = MotionAutoencoder(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # 3. Get latent vectors for a subset of the data
    print(f"Extracting latent vectors from {sample_size} samples...")
    all_latents = []
    all_labels = []
    
    # --- MAPPING FOR MOTION TYPES ---
    # This map links action codes from filenames to readable categories.
    # You may need to inspect your filenames and adjust or expand this map.
    ACTION_MAP = {
        '01': 'Walking',
        '02': 'Running/Jogging',
        '03': 'Jumping',
        '04': 'Basketball',
        '05': 'Soccer',
        '06': 'Boxing',
        '07': 'Waving/Gesturing',
        '08': 'Dancing',
        '09': 'Stretching/Yoga'
    }

    # Use a random subset of indices for variety
    indices = np.random.choice(len(dataset), sample_size, replace=False)

    for idx in tqdm(indices, desc="Encoding samples"):
        sample = dataset[idx]
        
        # --- MODIFIED LABELING LOGIC ---
        # Get the source file path to create a motion type label
        file_path, _ = dataset.windows[idx]
        filename = os.path.basename(file_path) # e.g., "02_01.bvh"
        
        # Parse the filename to get the action code (e.g., '01' from '02_01.bvh')
        try:
            action_code = filename.split('_')[1].split('.')[0]
            # Use the map to get a readable label, defaulting to 'Other' if not found
            label = ACTION_MAP.get(action_code, 'Other')
        except IndexError:
            label = 'Unknown' # Handle files with different naming schemes

        all_labels.append(label)
        
        # Prepare model input (same as in training)
        positions = sample["positions_normalized_flat"].unsqueeze(0).to(device)
        trans_vel = sample["trans_vel_xz"].unsqueeze(0).to(device)
        rot_vel = sample["rot_vel_y"].unsqueeze(0).to(device).unsqueeze(-1)
        model_input = torch.cat([positions, trans_vel, rot_vel], dim=2)

        # Get the latent vector 'z' from the encoder
        with torch.no_grad():
            _, z = model(model_input)
        
        # Flatten the latent vector and move to CPU
        z_flat = z.detach().cpu().numpy().flatten()
        all_latents.append(z_flat)

    # 4. Perform t-SNE dimensionality reduction
    print("Performing t-SNE reduction. This may take a few minutes...")
    tsne = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(np.array(all_latents))
    print("t-SNE completed.")

    # 5. Plot the results
    print(f"Plotting results and saving to {output_file}...")
    plt.figure(figsize=(14, 10))
    
    # Use pandas to easily map string labels to colors
    label_codes, unique_labels = pd.factorize(all_labels, sort=True)

    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=label_codes, cmap='tab20', alpha=0.8)
    
    plt.title('t-SNE Visualization of the Learned Motion Manifold', fontsize=16)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.2)
    
    # Create a legend with the new motion type labels
    plt.legend(handles=scatter.legend_elements(num=len(unique_labels))[0], labels=unique_labels.tolist(), title="Motion Type")

    plt.savefig(output_file)
    plt.close()
    print("✅ Visualization saved!")

if __name__ == "__main__":
    generate_tsne_visualization(
        model_path="./output/ae/models/motion_autoencoder.pt",
        data_dir="./cmu-mocap",
        output_file="./output/ae/plots/manifold_visualization_by_type.png"
    )