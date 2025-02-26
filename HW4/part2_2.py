from transformers import BertTokenizer, BertModel
from data import DiplomacyDataset
from torch.utils.data import DataLoader
import argparse
import torch
import torch.nn as nn
from transformers import AdamW

def collate_fn(data):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    inputs = tokenizer(sents, padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(labels)
    return inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"], labels


class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=768, out_features=2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained_model(input_ids, attention_mask, token_type_ids)
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=-1)
        return out

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_file", type=str, default="diplomacy_cv.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epoch_num", type=int, default=10)
    args = parser.parse_args()

    dataset = DiplomacyDataset(args)
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

    pretrained_model = BertModel.from_pretrained("bert-base-uncased")
    for params in pretrained_model.parameters():
        params.requires_grad = False

    model = ClassificationModel()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(args.epoch_num):
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
            output = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 5 == 0:
                out = output.argmax(dim=1)
                accuracy = (out == labels).sum().item() / len(labels)

                print(i, loss.item(), accuracy)

        

