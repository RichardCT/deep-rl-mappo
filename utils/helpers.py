import os
import numpy as np
import time

import random
import torch
import torch.nn.functional as F

def load_npy_files_to_dict(folder_path="./states/embedding/"):
    """
    Load all .npy files from a folder into memory and store them in a dictionary.
    
    Args:
    folder_path (str): Path to the folder containing .npy files.
    
    Returns:
    dict: Dictionary with filename as key and NumPy array as value.
    """
    
    file_dict = {}
    i = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            try:
                array = np.load(file_path)
                key = filename.split('.')[0]
                file_dict[key] = array
                i += 1
            except Exception as e:
                print(f"Error loading file {filename}: {str(e)}")
    print(f"Loaded {i} cached states into memory!")
    return file_dict

def get_run_name(config):
    exp_name = config['simulation']['name']
    run_name = f"{config['env']['id']}__{exp_name}__{config['simulation']['seed']}__{int(time.time())}"
    if not os._exists(f"runs/{run_name}"):
        os.makedirs(f"runs/{run_name}")
    return exp_name, run_name

def set_seed(seed, torch_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

def preprocess_observation(next_observation, img_size):
    """
    Preprocess the observation by reshaping, permuting, resizing, and normalizing.
    
    Args:
    next_observation (torch.Tensor): The observation tensor to preprocess.
    img_size (int): The size to which the image should be resized.
    
    Returns:
    torch.Tensor: The preprocessed observation tensor.
    """

    next_observation = next_observation.view(-1, img_size, img_size, 3)  # reshape
    next_observation = next_observation.permute(0, 3, 1, 2)  # change to [n, 3, img_size, img_size]

    # Resize to [n, 3, 128, 128]
    # _, _, y, _ = next_observation.shape
    # zoom_factors = (1, 1, 128 / y, 128 / y)
    # next_observation = zoom(next_observation, zoom_factors, order=1)  # order=1 for bilinear interpolation
    
    # Resize using bilinear interpolation
    next_observation = F.interpolate(next_observation, size=(128, 128), mode='bilinear', align_corners=False)
    
    # Squeeze batch dimension if input was single image
    # if next_observation.size(0) == 1:
    #     next_observation = next_observation.squeeze(0)

    next_observation = next_observation / 255.0  # normalize

    return next_observation
    
if __name__ == '__main__':
    folder_path = './states/embedding/'
    file_dict = load_npy_files_to_dict(folder_path)
    assert len(file_dict.keys()) > 0, "No files found in the folder."
    print(file_dict.keys())
    print(file_dict['0'])
    file_dict.get(f"{0}_{0}", np.zeros(384))