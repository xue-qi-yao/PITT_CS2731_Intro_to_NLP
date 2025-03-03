import torch
from torch.utils.data import Dataset, DataLoader
import re
import pandas as pd
import argparse
from pathlib import Path
import numpy as np

def text_processing(text):
    text = re.sub(r'[^\a-zA-Z0-9\s]', "", text).strip()
    return text

class DiplomacyTrainingDataset(Dataset):
    def __init__(self, args, fold_num):
        super().__init__()
        df = pd.read_csv(Path(args.training_file), index_col="id")
        df["text"] = df["text"].apply(text_processing)
        list_of_array_input = np.array_split(df["text"].to_numpy(), 5)
        list_of_array_label = np.array_split(df["intent"].to_numpy(), 5)
        list_of_input = np.concatenate([list_of_array_input[i] for i in range(5) if i != fold_num])
        list_of_label = np.concatenate([list_of_array_label[i] for i in range(5) if i != fold_num])
        self.list_of_text = list(list_of_input)
        self.labels = list(list_of_label)

    def __getitem__(self, index):
        text = self.list_of_text[index]
        label = self.labels[index]

        return text, label
    
    def __len__(self):
        return len(self.labels)

class DiplomacyValidationDataset(Dataset):
    def __init__(self, args, fold_num):
        super().__init__()
        df = pd.read_csv(Path(args.training_file), index_col="id")
        df["text"] = df["text"].apply(text_processing)
        list_of_array_input = np.array_split(df["text"].to_numpy(), 5)
        list_of_array_label = np.array_split(df["intent"].to_numpy(), 5)
        list_of_input = list_of_array_input[fold_num]
        list_of_label = list_of_array_label[fold_num]
        self.list_of_text = list(list_of_input)
        self.labels = list(list_of_label)

    def __getitem__(self, index):
        text = self.list_of_text[index]
        label = self.labels[index]

        return text, label
    
    def __len__(self):
        return len(self.labels)
    
class DiplomacyTestDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        df = pd.read_csv(Path(args.testing_file), index_col="id")
        df["text"] = df["text"].apply(text_processing)
        self.list_of_text = df["text"].to_list()

    def __getitem__(self, index):
        text = self.list_of_text[index]
        return text
    
    def __len__(self):
        return len(self.list_of_text)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_file", default="diplomacy_cv.csv")
    args = parser.parse_args()
    dataset = DiplomacyTrainingDataset(args)
    print(dataset[0], len(dataset))