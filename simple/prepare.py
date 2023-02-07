"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
from utils import download_data, read_data, split_data,Encoder
import numpy as np




work_dir = os.path.dirname(__file__)
input_file_path = os.path.join(work_dir, 'input.txt')
DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

# download the tiny shakespeare dataset
download_data(DATA_URL, input_file_path)
data = read_data(input_file_path)

print(f"length of dataset in characters: {len(data):,}")

chars = sorted(list(set(data)))
vocab_size = len(chars)
encoder = Encoder(data)

# create the train and test splits
train_data, val_data = split_data(data,ratio=0.9)

# encode both to integers
train_ids = encoder.encode(train_data)
val_ids = encoder.encode(val_data)

# convert to numpy array
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

# export to bin files
train_ids.tofile(os.path.join(work_dir, 'train.bin'))
val_ids.tofile(os.path.join(work_dir, 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': encoder.itos,
    'stoi': encoder.stoi, 
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
