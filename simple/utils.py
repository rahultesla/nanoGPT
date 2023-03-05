import os
import requests
import pickle

def download_data(url, file_path):
    if os.path.exists(file_path):
        return
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(requests.get(url,timeout=1000).text)

def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    return data

def path(filename):
    work_dir = os.path.dirname(__file__)
    return os.path.join(work_dir, filename)

def split_data(data,ratio):
    data_size = len(data)
    train_data = data[:int(data_size*ratio)]
    val_data = data[int(data_size*ratio):]
    return train_data,val_data

class Encoder():
    def __init__(self,data) -> None:
        chars = sorted(list(set(data)))
        self.stoi:dict[str,int] = {ch: i for i, ch in enumerate(chars)}
        self.itos:dict[int,str] = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.metadata = {
            'vocab_size': self.vocab_size,
            'itos': self.itos,
            'stoi': self.stoi,
        }
    def encode(self,text:str) -> list[int]:
        return [self.stoi[c] for c in text]
    def decode(self,encoded_data:list[int]) -> str:
        return ''.join([self.itos[i] for i in encoded_data])

    def save(self, filepath) -> None:
        with open(filepath, 'wb') as file:
            pickle.dump(self.metadata, file)


class TrainingParameters:

    def __init__(self) -> None:
        self.out_dir = 'out-shakespeare-char'
        self.eval_interval = 250    # keep frequent because we'll overfit
        self.eval_iters = 200
        self.log_interval = 10      # don't print too too often
        # we expect to overfit on this small dataset, so only save when val improves
        self.always_save_checkpoint = False
        self.wandb_log = False      # override via command line if you like
        self.wandb_project = 'shakespeare-char'
        self.wandb_run_name = 'mini-gpt'
        self.dataset = 'shakespeare_char'
        self.batch_size = 64
        self.block_size = 256       # context of up to 256 previous characters

        # baby GPT model :)
        self.n_layer = 6
        self.n_head = 6
        self.n_embd = 384
        self.dropout = 0.2

        self.learning_rate = 1e-3   # with baby networks can afford to go a bit higher
        self.max_iters = 5000
        self.lr_decay_iters = 5000  # make equal to max_iters usually
        self.min_lr = 1e-4          # learning_rate / 10 usually
        self.beta2 = 0.99           # make a bit bigger because number of tokens per iter is small
        self.warmup_iters = 100     # not super necessary potentially
        self.device = 'cpu'         # run on cpu only
        self.compile = False        # do not torch compile the model

# --device=cpu 
# --compile=False 
# --eval_iters=20 
# --log_interval=1 
# --block_size=64 
# --batch_size=12 
# --n_layer=4 
# --n_head=4 
# --n_embd=128 
# --max_iters=2000 
# --lr_decay_iters=2000 
# --dropout=0.0