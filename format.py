# python format.py
import os
import glob
import shutil
import random
from tqdm import tqdm
import sys
import traceback

# %% [markdown]
# ### 1. Configuration and Path Setup
# %%
# Source Directory: The original BraTS2020 dataset location
SOURCE_BASE_DIR = r"your_path/to/MICCAI_BraTS2020_TrainingData"

# Destination Directory: Where the split sets will be organized
DEST_BASE_DIR = r"your_path/to/github_brain_tumor"
TRAIN_DIR = os.path.join(DEST_BASE_DIR, "train_set")
VAL_DIR = os.path.join(DEST_BASE_DIR, "val_set")
TEST_DIR = os.path.join(DEST_BASE_DIR, "test_set")

# Data Split Ratios (80% Train, 10% Val, 10% Test)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1 

# Seed for deterministic and reproducible splitting
SEED = 2024

# %% [markdown]
# ### 2. File Integrity Utilities
# %%
def find_patient_folders(base_dir):
    """
    Scans the base directory for patient folders and verifies file integrity
    by checking for necessary modality files (.nii or .nii.gz).
    """
    print(f"Scanning directory: {base_dir}")
    patient_folders = sorted(
        [d for d in glob.glob(os.path.join(base_dir, "BraTS20_Training_*")) if os.path.isdir(d)]
    )
    
    clean_folders = []
    
    for folder_path in tqdm(patient_folders, desc="Checking file integrity"):
        ctid = os.path.basename(folder_path)
        
        # Required modalities for BraTS format
        required_names = [
            f"{ctid}_flair", f"{ctid}_t1", 
            f"{ctid}_t1ce", f"{ctid}_t2", f"{ctid}_seg"
        ]
        
        files_found = 0
        for required_name in required_names:
            if os.path.isfile(os.path.join(folder_path, required_name + ".nii")):
                files_found += 1
            elif os.path.isfile(os.path.join(folder_path, required_name + ".nii.gz")):
                files_found += 1
        
        # Only add folder if all 4 modalities + 1 segmentation are present
        if files_found == len(required_names):
            clean_folders.append(folder_path)
        else:
            print(f"Warning: Skipping {ctid} (Missing files: found {files_found}/{len(required_names)})")
            
    print(f"Total verified patient folders found: {len(clean_folders)}")
    return clean_folders

def copy_patient_folder(source_path, dest_dir):
    """
    Copies verified patient data to the destination split folder.
    Skips if the destination already exists and is complete.
    """
    folder_name = os.path.basename(source_path)
    dest_path = os.path.join(dest_dir, folder_name)
    
    if os.path.exists(dest_path) and len(os.listdir(dest_path)) >= 5: 
        return True 

    try:
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path) # Clean incomplete copy
        
        shutil.copytree(source_path, dest_path, copy_function=shutil.copy2)
        return True
    except Exception as e:
        print(f"Error copying {folder_name}: {e}")
        return False

# %% [markdown]
# ### 3. Dataset Splitting Logic
# %%
def main():
    # Find all complete data folders
    all_patient_folders = find_patient_folders(SOURCE_BASE_DIR)
    
    if not all_patient_folders:
        print("ERROR: No valid data found in Source Directory!")
        sys.exit(1)
        
    total_count = len(all_patient_folders)
    
    # Perform deterministic shuffle
    random.seed(SEED)
    random.shuffle(all_patient_folders) 

    # Partitioning the list
    val_count = int(total_count * VAL_RATIO) 
    test_count = int(total_count * TEST_RATIO) 
    train_count = total_count - val_count - test_count

    train_paths = all_patient_folders[:train_count]
    val_paths = all_patient_folders[train_count : train_count + val_count]
    test_paths = all_patient_folders[train_count + val_count :]

    print("\n--- Data Split Summary ---")
    print(f"Total Cases: {total_count}")
    print(f"Train Set: {len(train_paths)} ({len(train_paths)/total_count:.1%})")
    print(f"Validation Set: {len(val_paths)} ({len(val_paths)/total_count:.1%})")
    print(f"Test Set: {len(test_paths)} ({len(test_paths)/total_count:.1%})")
    
    # Create split directories
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # Execute file copying for each set
    print("\nProcessing Train Set...")
    for path in tqdm(train_paths, desc="Copying Train"):
        copy_patient_folder(path, TRAIN_DIR)

    print("\nProcessing Validation Set...")
    for path in tqdm(val_paths, desc="Copying Val"):
        copy_patient_folder(path, VAL_DIR)
        
    print("\nProcessing Test Set...")
    for path in tqdm(test_paths, desc="Copying Test"):
        copy_patient_folder(path, TEST_DIR)

    print("\nData organization complete!")
    print(f"Output located at: {DEST_BASE_DIR}")

# %% [markdown]
# ### 4. Execution Entry Point
# %%
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        traceback.print_exc(file=sys.stdout)