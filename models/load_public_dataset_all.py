
from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import random

def robust_minmax(x, pmin=1, pmax=99):
    """Percentile-based min-max normalization, avoids outliers."""
    import numpy as np
    lo, hi = np.percentile(x, [pmin, pmax])
    x = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    return x


def split_and_label(final_pet_lists, train_ratio=0.8):
    train_set = []
    test_set = []

    for pet_list in final_pet_lists:
        random.shuffle(pet_list)  # Shuffle the list to ensure randomness
        split_idx = int(len(pet_list) * train_ratio)
        train_set.extend(pet_list[:split_idx])
        test_set.extend(pet_list[split_idx:])

    return train_set, test_set


def generate_nifti_structure_test(folder_path, structural_mri, pet_images):
    """
    Build a combined list across ALL subject subfolders:
      [mri_list, pet_path, pet_index]
    No train/test split; used for sampling/inference.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Base data folder does not exist: {folder_path}")

    # Subject subfolders, e.g., /content/data/Data/941_S_10065
    subfolders = [
        os.path.join(folder_path, sub)
        for sub in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, sub))
    ]

    combined = []
    num_subjects = 0
    num_entries = 0

    for subfolder in subfolders:
        try:
            files = os.listdir(subfolder)
        except Exception as e:
            print(f"[WARN] Could not list: {subfolder} ({e})")
            continue

        num_subjects += 1

        # MRI list in the order provided by structural_mri
        mri_list = [
            os.path.join(subfolder, file) if file in files else ''
            for file in structural_mri
        ]

        # For each PET tracer present, append an entry
        for pet in pet_images:
            if pet in files:
                combined.append([mri_list, os.path.join(subfolder, pet), pet_images.index(pet)])
                num_entries += 1

    if num_entries == 0:
        raise RuntimeError(
            "No matching PET files were found under "
            f"{folder_path}. Check modality names in config and folder contents."
        )

    print(f"[INFO] Test build: subjects scanned={num_subjects}, entries created={num_entries}")
    return combined

def generate_nifti_structure(folder_path, structural_mri, pet_images):
    """
    Build per-tracer lists of [mri_list, pet_path, pet_index] across subject subfolders,
    then split each tracer list into train/test and merge.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Base data folder does not exist: {folder_path}")

    subfolders = [
        os.path.join(folder_path, sub)
        for sub in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, sub))
    ]

    # Collect entries per tracer
    pet_types = {}  # tracer_name -> list of [mri_list, pet_path, pet_index]
    for subfolder in subfolders:
        try:
            files = os.listdir(subfolder)
        except Exception as e:
            print(f"[WARN] Could not list: {subfolder} ({e})")
            continue

        # MRI list: keep order, allow missing with ''
        mri_list = [
            os.path.join(subfolder, f) if f in files else ''
            for f in structural_mri
        ]

        # For each PET tracer present, append an entry
        for pet in pet_images:
            if pet in files:
                pet_types.setdefault(pet, []).append(
                    [mri_list, os.path.join(subfolder, pet), pet_images.index(pet)]
                )

    # Merge non-empty tracer lists and split each into train/test
    final_pet_lists = [lst for lst in pet_types.values() if lst]
    train_set, test_set = split_and_label(final_pet_lists, train_ratio=0.8)
    return train_set, test_set


# TRAIN dataset (3 items): inputs, label, pet_type
class MyDataset(Dataset):
    """
    TRAIN dataset.
    Returns exactly (inputs, label, pet_type)
      - inputs: (num_modalities, Z, Y, X) float32 tensor
      - label:  (Z, Y, X) float32 tensor (PET ground truth)
      - pet_type: int tensor (index of tracer in pet_images)
    """
    def __init__(self, dataset):
        # dataset = list of [mri_list, pet_path, pet_index]
        self.dataset = dataset
    def __getitem__(self, index):
        mri_list, pet_path, pet_index = self.dataset[index]
        pet_type = torch.tensor(pet_index)

        # Load PET label (target)
        labels_nii = nib.load(pet_path)
        label_np = labels_nii.get_fdata().astype(np.float32)
        label_np = robust_minmax(label_np)   # replaced normalization
        label = torch.from_numpy(label_np)

        # Load MRI inputs; replace missing paths with zeros matching PET shape
        input_np_list = []
        for mri_path in mri_list:
            if not mri_path:
                input_np_list.append(np.zeros_like(label_np, dtype=np.float32))
            else:
                mri_nii = nib.load(mri_path)
                mri_np = mri_nii.get_fdata().astype(np.float32)
                mri_np = robust_minmax(mri_np)  # same normalization for MRI
                input_np_list.append(mri_np)

        inputs = torch.from_numpy(np.array(input_np_list, dtype=np.float32))  # (M, Z, Y, X)
        return inputs, label, pet_type

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def zscore_brain(arr, eps=1e-6):
        m = np.mean(arr)
        s = np.std(arr)
        if s < eps:  # avoid div by zero
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - m) / (s + eps)).astype(np.float32)

    @staticmethod
    def min_max_normalize(arr):
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        if max_val == 0.0:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - min_val) / (max_val - min_val)).astype(np.float32)

    @staticmethod
    def get_brain_mask(volume, threshold=0.05):
        """Simple binary mask for non-background voxels."""
        return (volume > threshold * volume.max()).astype(np.float32)


class MyDatasetTest(Dataset):
    """
    Test-time dataset that also returns metadata for manifest writing:
    returns: inputs, label, pet_type, pet_path, mri_list
    """
    def __init__(self, dataset):
        self.dataset = dataset  # list of [mri_list, pet_path, pet_index]

    def __getitem__(self, index):
        mri_list, pet_path, pet_index = self.dataset[index]
        pet_type = torch.tensor(pet_index)

        # Load PET label volume
        labels_nii = nib.load(pet_path)
        Mask_numpy = labels_nii.get_fdata().astype(np.float32)
        Mask_numpy = robust_minmax(Mask_numpy)  # replaced normalization
        label = torch.from_numpy(Mask_numpy)
        
        input_numpy_list = []
        for p in mri_list:
            if len(p) == 0:
                input_numpy_list.append(np.zeros_like(Mask_numpy))
            else:
                CT_nii = nib.load(p)
                CT_numpy = CT_nii.get_fdata().astype(np.float32)
                CT_numpy = robust_minmax(CT_numpy)  # replaced normalization
                input_numpy_list.append(CT_numpy)


        input_numpy_list = np.array(input_numpy_list)  # (num_modalities, Z, Y, X)
        inputs = torch.from_numpy(input_numpy_list)

        # Return metadata so the sampler can write a manifest
        return inputs, label, pet_type, pet_path, mri_list

    def __len__(self):
        return len(self.dataset)


def load_data(structural_mri, pet_images, folder_path, train=True, num_workers=0, pin_memory=False, persistent_workers=False):
    folder_path = folder_path  # path comes from config

    train_set, test_set = generate_nifti_structure(folder_path, structural_mri, pet_images)

    if train:
        my_dataset = MyDataset(train_set)
    else:
        # Use the config-provided folder/modality lists across ALL subjects
        final_pet_lists = generate_nifti_structure_test(folder_path, structural_mri, pet_images)
        my_dataset = MyDatasetTest(final_pet_lists)

    loader = DataLoader(
        my_dataset, batch_size=1, shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=False
    )
    return loader
    

def find_folders_with_specific_files(folder_path, required_files):
    # List all subdirectories in the folder
    subfolders = [os.path.join(folder_path, subfolder) for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]

    matching_folders = []

    for subfolder in subfolders:
        # List all files in the subfolder
        files = os.listdir(subfolder)

        # Check if all required files are in the subfolder
        if all(file in files for file in required_files):
            matching_folders.append(subfolder)

    return matching_folders










