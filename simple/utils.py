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
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
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

