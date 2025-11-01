import torch
import numpy as np
import os
import random
from tqdm import tqdm

# Import the necessary classes and functions from your existing files
from dataloader import CMUMotionDataset
from AE import MotionAutoencoder, MotionManifoldSynthesizer
from visualization import visualize_interpolation, visualize_motion_comparison
from visualization import visualize_style_transfer

'''
 Available joint names: ['Hips', 'LHipJoint', 'LeftUpLeg', 
 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'End Site', 'RHipJoint', 
 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'End Site', 
 'LowerBack', 'Spine', 'Spine1', 'Neck', 'Neck1', 'Head', 'End Site', 
 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftFingerBase', 
 'LeftHandIndex1', 'End Site', 'LThumb', 'End Site', 'RightShoulder', 'RightArm', 
 'RightForeArm', 'RightHand', 'RightFingerBase', 'RightHandIndex1', 'End Site', 
 'RThumb', 'End Site']
'''

def generate_motion_examples():
    """
    Main function to load the model and generate interpolation/corruption video examples.
    """
    
    # Adjust these parameters as needed
    MODEL_PATH = "./output/ae/models/motion_autoencoder.pt"
    DATA_DIR = "./cmu-mocap"
    OUTPUT_DIR = "./generated_examples"
    NUM_INTERPOLATION_VIDEOS = 5
    NUM_CORRUPTION_VIDEOS = 5

    print("Starting Motion Example Generator...")

    # Setup paths and device
    os.makedirs(os.path.join(OUTPUT_DIR, "interpolations"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "corruptions"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "completions"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Dataset
    print("Loading dataset...")
    dataset = CMUMotionDataset(
        data_dir=DATA_DIR,
        frame_rate=30,
        window_size=160,
        overlap=0.5
    )
    print(f"Dataset loaded with {len(dataset)} total motion windows.")

    # 2. Load Synthesizer
    print(f"Loading model from {MODEL_PATH}...")
    try:
        synthesizer = MotionManifoldSynthesizer(
            model_path=MODEL_PATH,
            dataset=dataset,
            device=device
        )
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please make sure you have trained the model and the path is correct.")
        return
    except Exception as e:
        print(f"An error occurred loading the model: {e}")
        print("Please ensure your AE.py and dataloader.py files are up to date.")
        return
        
    print("Synthesizer loaded successfully.")

    # 3. Define Corruption Types
    # We define the types of corruption to apply
    corruption_types = []
    corruption_types.append({'name': 'zero_noise_40pct', 'type': 'zero', 'params': {'prob': 0.4}})
    corruption_types.append({'name': 'gaussian_noise', 'type': 'noise', 'params': {'scale': 0.1}})
    
    # Safely find a joint to remove (e.g., 'LeftForeArm')
    try:
        joint_to_remove_idx = synthesizer.joint_names.index('LeftForeArm')
        corruption_types.append({'name': 'missing_lForeArm', 'type': 'missing', 'params': {'joint_idx': joint_to_remove_idx}})
    except ValueError:
        print("Warning: 'LeftForeArm' joint not found. Skipping 'missing_limb' corruption.")
        
    try:
        joint_to_remove_idx = synthesizer.joint_names.index('LeftFoot')
        corruption_types.append({'name': 'missing_lFoot', 'type': 'missing', 'params': {'joint_idx': joint_to_remove_idx}})
    except ValueError:
        print("Warning: 'LeftFoot' joint not found. Skipping 'missing_limb' corruption.")

    # # --- GENERATE INTERPOLATION VIDEOS ---
    # print(f"\nGenerating {NUM_INTERPOLATION_VIDEOS} interpolation videos...")
    # interp_dir = os.path.join(OUTPUT_DIR, "interpolations")
    # for i in tqdm(range(NUM_INTERPOLATION_VIDEOS), desc="Interpolations"):
    #     try:
    #         # Get two different random motion indices
    #         idx1, idx2 = random.sample(range(len(dataset)), 2)
            
    #         motion1 = dataset[idx1]
    #         motion2 = dataset[idx2]

    #         # Perform the interpolation (t=0.5 for a 50/50 blend)
    #         interpolated_motion = synthesizer.interpolate_motions(motion1, motion2, t=0.5)

    #         # Get the original local motions for visualization
    #         motion1_local = motion1['positions']
    #         motion2_local = motion2['positions']

    #         # Define output path
    #         output_path = os.path.join(interp_dir, f"interp_{i+1:02d}_({idx1}_vs_{idx2}).mp4")

    #         # Create the side-by-side-by-side video
    #         visualize_interpolation(motion1_local, motion2_local, interpolated_motion, synthesizer.joint_parents, output_path)
        
    #     except Exception as e:
    #         print(f"Error generating interpolation video {i+1}: {e}")

    # # --- GENERATE CORRUPTION VIDEOS ---
    # print(f"\nGenerating {NUM_CORRUPTION_VIDEOS} corruption video sets...")
    # corrupt_dir = os.path.join(OUTPUT_DIR, "corruptions")
    # for i in tqdm(range(NUM_CORRUPTION_VIDEOS), desc="Corruptions"):
    #     try:
    #         # Get one random motion index
    #         idx = random.randint(0, len(dataset) - 1)
    #         sample_motion = dataset[idx]

    #         # Apply and fix each type of corruption to this one motion
    #         for corruption in corruption_types:
    #             corrupted, fixed = synthesizer.fix_corrupted_motion(
    #                 sample_motion,
    #                 corruption['type'],
    #                 corruption['params']
    #             )

    #             # Define output path
    #             output_path = os.path.join(corrupt_dir, f"corr_sample_{i+1:02d}_({idx})_{corruption['name']}.mp4")

    #             # Create the side-by-side comparison video
    #             visualize_motion_comparison(corrupted, fixed, synthesizer.joint_parents, output_path)

    #     except Exception as e:
    #         print(f"Error generating corruption video set {i+1}: {e}")

    # # --- GENERATE MOTION COMPLETION VIDEOS (EXTRA CREDIT) ---
    # print(f"\nGenerating Motion Completion (Extra Credit) videos...")
    # comp_dir = os.path.join(OUTPUT_DIR, "completions")
    # window_size = dataset.window_size # Should be 160
    
    # # --- 1. In-painting (Filling Gaps) ---
    # print(f"  Generating {NUM_INTERPOLATION_VIDEOS} in-painting (gap filling) videos...")
    
    # # Create a mask to keep the start and end, but remove the middle
    # gap_size = window_size // 2 # 80 frames
    # keep_size = window_size // 4 # 40 frames
    
    # inpaint_mask = torch.zeros(window_size)
    # inpaint_mask[:keep_size] = 1.0  # Keep first 40 frames
    # inpaint_mask[window_size - keep_size:] = 1.0 # Keep last 40 frames
    
    # for i in tqdm(range(NUM_INTERPOLATION_VIDEOS), desc="In-painting"):
    #     try:
    #         idx = random.randint(0, len(dataset) - 1)
    #         sample_motion = dataset[idx]
            
    #         # Call the new completion function
    #         masked_viz, completed_motion = synthesizer.complete_partial_motion(
    #             sample_motion, 
    #             inpaint_mask,
    #             num_iterations=200,
    #             transition_weight=1.0,
    #             foot_contact_weight=2.0,
    #             smoothness_weight=0.5
    #         )
            
    #         output_path = os.path.join(comp_dir, f"inpaint_{i+1:02d}_(gap_frames_{keep_size}-{window_size-keep_size}).mp4")
    #         visualize_motion_comparison(masked_viz, completed_motion, synthesizer.joint_parents, output_path)
            
    #     except Exception as e:
    #         print(f"Error generating in-painting video {i+1}: {e}")
            
    # # --- 2. Out-painting (Extending Sequences) ---
    # print(f"\n  Generating {NUM_INTERPOLATION_VIDEOS} out-painting (extension) videos...")
    
    # # Create a mask to keep only the first half
    # keep_frames = window_size // 2 # Keep first 80 frames
    # outpaint_mask = torch.zeros(window_size)
    # outpaint_mask[:keep_frames] = 1.0
    
    # for i in tqdm(range(NUM_INTERPOLATION_VIDEOS), desc="Out-painting"):
    #     try:
    #         idx = random.randint(0, len(dataset) - 1)
    #         sample_motion = dataset[idx]
            
    #         # Call the new completion function
    #         masked_viz, completed_motion = synthesizer.complete_partial_motion(
    #             sample_motion, 
    #             outpaint_mask,
    #             num_iterations=200,
    #             transition_weight=1.0,
    #             foot_contact_weight=2.0,
    #             smoothness_weight=0.5
    #         )
            
    #         output_path = os.path.join(comp_dir, f"outpaint_{i+1:02d}_(extend_from_frame_{keep_frames}).mp4")
    #         visualize_motion_comparison(masked_viz, completed_motion, synthesizer.joint_parents, output_path)
            
    #     except Exception as e:
    #         print(f"Error generating out-painting video {i+1}: {e}")
            
    # --- GENERATE STYLE TRANSFER VIDEOS (EXTRA CREDIT) ---
    print(f"\nGenerating Motion Style Transfer (Extra Credit) videos...")
    style_dir = os.path.join(OUTPUT_DIR, "style_transfer")
    os.makedirs(style_dir, exist_ok=True)
    NUM_STYLE_VIDEOS = 3 # As requested by the assignment

    for i in tqdm(range(NUM_STYLE_VIDEOS), desc="Style Transfer Videos"):
        try:
            # Get two different random motion indices for content and style
            content_idx, style_idx = random.sample(range(len(dataset)), 2)
            content_motion_sample = dataset[content_idx]
            style_motion_sample = dataset[style_idx]

            # Perform the style transfer
            result_motion = synthesizer.transfer_style(
                content_motion_sample,
                style_motion_sample,
                num_iterations=250,      # Can be tuned for quality vs. speed
                learning_rate=0.01,
                content_weight=1.0,      # As recommended in paper
                style_weight=0.5        # As recommended in paper
            )

            # Get original local motions for visualization
            content_motion_local = content_motion_sample['positions']
            style_motion_local = style_motion_sample['positions']

            # Define the output path
            output_path = os.path.join(style_dir, f"style_transfer_{i+1:02d}_(content_{content_idx}_style_{style_idx}).mp4")

            # Create the three-panel visualization
            # This function is in visualization.py and reuses the interpolation visualizer
            visualize_style_transfer(
                content_motion_local, # Displayed on the left
                style_motion_local,   # Displayed on the right
                result_motion,        # Displayed in the middle
                synthesizer.joint_parents,
                output_path
            )
        except Exception as e:
            print(f"Error generating style transfer video {i+1}: {e}")
    
    print(f"\n All examples saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_motion_examples()
