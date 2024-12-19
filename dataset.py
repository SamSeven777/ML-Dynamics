#!/home/runyuq2/anaconda3/envs/torch1.12/bin/python
import numpy as np
import torch

def parse_pdb_file(filename, num_models, num_residues):
    features = np.zeros((num_models, num_residues, 3), dtype=np.float32)
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} not fount.")

    current_model = None
    for line in lines:
        if line.startswith("MODEL"):
            current_model = int(line.split()[1]) - 1
        elif line.startswith("ENDMDL"):
            current_model = None
        elif current_model is not None and line.startswith("ATOM"):
            try:
                residue = int(line[22:26].strip()) - 1
                features[current_model, residue, :] = float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())
            except (IndexError, ValueError):
                continue

    return features
    
def create_dataset(features):
    num_models, num_residues, _ = features.shape
    dataset = np.zeros((num_models - 1, num_residues, 6), dtype=np.float32)
    dataset[:,:,:3] = 0.5 * (features[1:,:,:] + features[:-1,:,:])
    dataset[:,:,3:] = features[1:,:,:] - features[:-1,:,:]

    loc_max = np.max(dataset[:,:,:3])
    loc_min = np.min(dataset[:,:,:3])
    vel_max = np.max(dataset[:,:,3:])
    vel_min = np.min(dataset[:,:,3:])
    
    dataset[:,:,:3] = (dataset[:,:,:3] - loc_min) * 2 / (loc_max - loc_min) - 1
    dataset[:,:,3:] = (dataset[:,:,3:] - vel_min) * 2 / (vel_max - vel_min) - 1
    
    return dataset
    
####
features = parse_pdb_file(filename="p12000_run2_clone4_CA_trimmed_aligned.pdb", num_models=8000, num_residues=334)

data_array = create_dataset(features)
np.save("data.npy", data_array)
