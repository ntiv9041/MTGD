from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import random



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
        Mask_numpy = labels_nii.get_fdata()
        Mask_numpy = MyDatasetTest.min_max_normalize(self=None, arr=Mask_numpy)  # reuse static method
        label = torch.from_numpy(Mask_numpy)

        # Load MRI condition volumes
        input_numpy_list = []
        for p in mri_list:
            if len(p) == 0:
                input_numpy_list.append(np.zeros_like(Mask_numpy))
            else:
                CT_nii = nib.load(p)
                CT_numpy = CT_nii.get_fdata()
                CT_numpy = MyDatasetTest.min_max_normalize(self=None, arr=CT_numpy)
                input_numpy_list.append(CT_numpy)

        input_numpy_list = np.array(input_numpy_list)  # (num_modalities, Z, Y, X)
        inputs = torch.from_numpy(input_numpy_list)

        # Return metadata so the sampler can write a manifest
        return inputs, label, pet_type, pet_path, mri_list

    def __len__(self):
        return len(self.dataset)

    def min_max_normalize(self,arr):

        min_val = np.min(arr)
        max_val = np.max(arr)

        normalized_arr = (arr - min_val) / (max_val - min_val)

        if max_val == 0:
          return np.zeros_like(arr)
        return normalized_arr


# required_files = [
#     '18F-AV45_ISO_BFC_rigid_normalised_brain.nii.gz',
#     '18F-AV1451_ISO_BFC_rigid_normalised_brain.nii.gz'
# ]
# mri = ['T2_ISO_BFC_normalised_brain.nii.gz', 'FLAIR_ISO_BFC_rigid_normalised_brain.nii.gz','T1_ISO_BFC_rigid_normalised_brain.nii.gz']
# path = "/g/data/iq24/public_PET/PET_preprosessing/MRI/MRI_dataset/ADNI/021_S_10114"



def load_data(structural_mri, pet_images, folder_path, train=True):
    folder_path = folder_path  # path comes from config

    train_set, test_set = generate_nifti_structure(folder_path, structural_mri, pet_images)

    if train:
        my_dataset = MyDatasetTest(train_set)
    else:
        # Use the config-provided folder/modality lists across ALL subjects
        final_pet_lists = generate_nifti_structure_test(folder_path, structural_mri, pet_images)
        my_dataset = MyDatasetTest(final_pet_lists)

    loader = DataLoader(
        my_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False
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
#folder_path = "/g/data/iq24/public_PET/PET_preprosessing/MRI/MRI_dataset/ADNI/"
#matching_folders = find_folders_with_specific_files(folder_path, required_files)
#print("Folders with required files:", matching_folders)



