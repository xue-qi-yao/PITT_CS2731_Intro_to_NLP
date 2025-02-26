import torch
from torch.utils.data import Dataset, DataLoader
import re
import pandas as pd
import argparse
from pathlib import Path

def text_processing(text):
    text = re.sub(r'[^\a-zA-Z0-9\s]', "", text).strip()
    return text

class DiplomacyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        df = pd.read_csv(Path(args.training_file), index_col="id")
        df["text"] = df["text"].apply(text_processing)
        self.list_of_text = list(df["text"])
        self.labels = list(df["intent"])

    def __getitem__(self, index):
        text = self.list_of_text[index]
        label = self.labels[index]

        return text, label
    
    def __len__(self):
        return len(self.labels)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_file", default="diplomacy_cv.csv")
    args = parser.parse_args()
    dataset = DiplomacyDataset(args)
    print(dataset[0], len(dataset))