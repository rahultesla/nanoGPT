"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""


from utils import download_data, read_data, split_data,path,Encoder
import numpy as np


input_file_path = path('input.txt')
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
train_ids.tofile(path('train.bin'))
val_ids.tofile(path('val.bin'))

# save the meta information as well, to help us encode/decode later
encoder.save(path('meta.pkl'))
