# Developer Information:
"""
## Developer Profile:
- Name: Vahid Seydi
- GitHub: https://github.com/vahidseydi

## Data Processing Functions:
- Description: Functions for preparing data and creating data loaders.
"""

# Importing necessary libraries
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from copy import deepcopy
import torch
import numpy as np

def toDataLoader(data, feature):
    """
    Converts data into a DataLoader.

    Args:
    - data (pd.DataFrame): Input data with features and labels.
    - feature (str): Name of the feature column.

    Returns:
    - data_loader (DataLoader): DataLoader for the input data.
    """

    # Extract features and labels
    X = data[feature].tolist()
    y = data['class'].tolist()

    # Convert to PyTorch tensors
    X = torch.tensor(np.array(X))
    y = torch.tensor(np.array(y))

    # Create a TensorDataset
    dataset = TensorDataset(X, y)

    # Create a DataLoader
    data_loader = DataLoader(dataset,  batch_size=len(
        dataset), shuffle=False, drop_last=False, num_workers=0)

    return data_loader

def toSequences(data, input_size):
    """
    Reshapes waveforms in the data to sequences.

    Args:
    - data (pd.DataFrame): Input data with waveforms.
    - input_size (int): Size of the input sequences.

    Returns:
    - df (pd.DataFrame): Dataframe with reshaped waveforms.
    """

    # Create a deep copy of the input data
    df = deepcopy(data)

    # Find the maximum length of waveforms
    max_wave_length = max([len(w) for w in data['wave']])

    # Find the correct reshaping factor
    for reshapeCorrector in range(0, max_wave_length):
        if (max_wave_length+reshapeCorrector) % input_size == 0:
            break

    # Calculate the sequence length
    seq_length = (max_wave_length+reshapeCorrector) // input_size

    # Reshape waveforms in the data
    for index, row in data.iterrows():
        x = np.pad(row['wave'], [0, reshapeCorrector],
                   mode='constant', constant_values=0)
        r_x = np.reshape(x, newshape=[seq_length, input_size])
        df.at[index, 'wave'] = r_x

    return df
