from transformers import BertTokenizer, BertModel
from data import DiplomacyTrainingDataset, DiplomacyValidationDataset
from torch.utils.data import DataLoader
import argparse
import torch
import torch.nn as nn
from transformers import AdamW
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def collate_fn(data):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    inputs = tokenizer(sents, padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(labels)
    return inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"], labels


class ClassificationModel(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.fc1 = nn.Linear(in_features=768, out_features=192)
        self.fc2 = nn.Linear(in_features=192, out_features=48)
        self.fc3 = nn.Linear(in_features=48, out_features=2)
        self.pretrained_model = pretrained_model

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
          out = pretrained_model(input_ids, attention_mask, token_type_ids)
        out = self.fc1(out.last_hidden_state[:, 0])
        out = self.fc2(out)
        out = self.fc3(out)
        out = out.softmax(dim=-1)
        return out

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_file", type=str, default="diplomacy_cv.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epoch_num", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    pretrained_model = BertModel.from_pretrained("bert-base-uncased").to(args.device)
    for parameter in pretrained_model.parameters():
        parameter.requires_grad_(False)

    model = ClassificationModel(pretrained_model).to(args.device)

    optimizer = AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    fold_loss_list = []
    fold_acc_list = []
    for fold_num in range(5):
        training_dataset = DiplomacyTrainingDataset(args, fold_num=fold_num)
        training_loader = DataLoader(dataset=training_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
        val_dataset = DiplomacyValidationDataset(args, fold_num=fold_num)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

        model.train()
        for epoch in range(args.epoch_num):
            train_loss_sum = 0
            train_acc_sum = 0 
            for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(training_loader):
                input_ids, attention_mask, token_type_ids, labels = input_ids.to(args.device), attention_mask.to(args.device), token_type_ids.to(args.device), labels.to(args.device)
                output = model(input_ids, attention_mask, token_type_ids)
                train_loss = loss_fn(output, labels)
                train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss_sum += train_loss.item()
                output = output.argmax(dim=-1)
                train_acc_sum += (output==labels).sum() / len(output)

            train_loss_result = train_loss_sum / len(training_loader)
            train_acc_result = train_acc_sum / len(training_loader)
            print(f"epoch {epoch} | loss {train_loss_result} | acc {train_acc_result}")

        model.eval()
        val_loss_sum = 0
        val_acc_sum = 0
        all_preds = []
        all_labels = []
        with torch.inference_mode():
            for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(val_loader):
                input_ids, attention_mask, token_type_ids, labels = input_ids.to(args.device), attention_mask.to(args.device), token_type_ids.to(args.device), labels.to(args.device)
                output = model(input_ids, attention_mask, token_type_ids)
                loss = loss_fn(output, labels)
                val_loss_sum += loss.item()
                output = output.argmax(dim=-1)
                val_acc_sum += (output==labels).sum() / len(output)

            val_loss_result = val_loss_sum / len(val_loader)
            val_acc_result = val_acc_sum / len(val_loader)
            all_preds.extend(output.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        print(f"fold {fold_num+1} | loss {val_loss_result} | acc {val_acc_result}")
        fold_loss_list.append(val_loss_result)
        fold_acc_list.append(val_acc_result)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"acc {acc} | precision {precision} | recall {recall} | f1-score {f1}")

        