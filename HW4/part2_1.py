import pandas as pd
import numpy as np
import re
import argparse

def text_processing(text):
    text = re.sub(r'[^\a-zA-Z0-9\s]', "", text).strip()
    text = re.split(r"[^\w']+", text)
    text = [word for word in text if word]
    return text

def text_to_feature_vector(list_of_text, word2idx, args):
    ndim = len(word2idx)
    n = len(list_of_text)
    feature_vectors = np.zeros((n, ndim))
    if args.binary:
        for r_idx, text_list in enumerate(list_of_text):
            word_set = set(text_list)
            for word in word_set:
                c_idx = word2idx[word]
                feature_vectors[r_idx, c_idx] = 1
    else:
        for r_idx, text_list in enumerate(list_of_text):
            for word in text_list:
                c_idx = word2idx[word]
                feature_vectors[r_idx, c_idx] += 1
    return feature_vectors

def train_preprocess(args):    
    df = pd.read_csv(args.training_file, index_col="id")
    df["text"] = df["text"].apply(text_processing)
    list_of_text = list(df["text"])
    print(f"number of text (n): {len(list_of_text)}")
    word_list, word_count = np.unique([word for list_of_word in list_of_text for word in list_of_word], return_counts=True)
    word2idx = dict(zip(word_list, range(len(word_list))))
    print(f"unique word number (ndim): {len(word2idx)}")

    y = np.array(df["intent"])[:, None]
    print(f"intent label (y) shape (y.shape==(n, 1)): {y.shape}")

    df = pd.read_csv(args.training_file, index_col="id")
    bow_matrix = text_to_feature_vector(list_of_text, word2idx, args)
    if args.tf_idf:
        tf_idf = np.log(word_count + 1)
        print(f"tf-idf vector (ndim): {tf_idf.shape[0]}")
        x = np.array(bow_matrix) * tf_idf
        print(f"feature vector (X) shape (X.shape==(n, ndim)): {x.shape}")
        return x, y, word_list, word2idx, tf_idf
    else: 
        x = np.array(bow_matrix)
        print(f"feature vector (X) shape (X.shape==(n, ndim)): {x.shape}")
        return x, y, word_list, word2idx


def test_preprocess(args, word_list, word2idx, tf_idf=None):
    df = pd.read_csv(args.testing_file, index_col="id")
    df["text"] = df["text"].apply(text_processing)
    list_of_text = list(df["text"])
    print(f"number of text (n): {len(list_of_text)}")
    x = np.zeros((len(list_of_text), len(word_list)))
    y = np.array(df["intent"])[:, None]

    if args.binary:
        for i, list_of_word in enumerate(list_of_text):
            unique_list_of_word = set(list_of_word)
            for word in unique_list_of_word:
                if word in word_list:
                    x[i, word2idx[word]] = 1
    else:
        for i, list_of_word in enumerate(list_of_text):
            for word in list_of_word:
                if word in word_list:
                    x[i, word2idx[word]] += 1
    if args.tf_idf:
        x *= tf_idf
    return x, y

class logistic_regression():
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.w = np.zeros((self.y.shape[1], self.x.shape[1]))
        self.b = np.zeros((self.y.shape[1], 1))
        self.fold_acc = []
    
    def cal_loss(self, pred, label):
        return -np.sum((label * np.log(pred) + (1-label) * np.log(1-pred))) / label.shape[0]
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train_(self, fold):
        print(f"training on fold {fold+1}")
        for epoch in range(self.epoch_num):
            logit = self.w @ self.train_x.T + self.b
            pred = self.sigmoid(logit)
            # loss = self.cal_loss(pred, self.train_y.T)
            self.w -= self.lr / self.train_x.shape[0] * np.sum((pred - self.train_y.T) * self.train_x.T, axis=-1)
            self.b -= self.lr / self.train_x.shape[0] * np.sum((pred - self.train_y.T), axis=-1)
            # print(f"fold {fold+1} | epoch {epoch} | loss {loss}")
    
    def val_(self, fold):
        logit = self.w @ self.val_x.T + self.b
        pred = np.where(self.sigmoid(logit)>=0.5, 1, 0)
        acc = np.sum(pred==self.val_y.T) / len(self.val_x)
        self.fold_acc.append(acc)
        print(f"fold {fold+1} | val acc {acc}")

    def test(self, x, y):
        logit = self.w @ x.T + self.b
        pred = np.where(self.sigmoid(logit)>=0.5, 1, 0)
        acc = np.sum(pred==y.T) / len(x)
        print(f"total acc {acc}")
        return pred

    def train(self, args):
        self.epoch_num = args.epoch
        self.lr = args.lr
        train_x_split = np.array_split(self.x, 5)
        train_y_split = np.array_split(self.y, 5)
        for i in range(5):
            self.train_x = np.concatenate([train_x_seg for j, train_x_seg in enumerate(train_x_split) if j != i], axis=0)
            self.train_y = np.concatenate([train_y_seg for j, train_y_seg in enumerate(train_y_split) if j != i], axis=0)
            self.train_(i)
            self.val_x = train_x_split[i]
            self.val_y = train_y_split[i]
            self.val_(i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default=False, type=bool, help="True for using binary bag-of-word")
    parser.add_argument("--tf_idf", default=False, type=bool, help="True for using the tf-idf transform")
    parser.add_argument("--epoch", default=1000, type=int, help="epoch number for each fold")
    parser.add_argument("--lr", default=100, type=int, help="learning rate for training")
    parser.add_argument("--training_file", default="diplomacy_cv.csv", type=str, help="training file path")
    parser.add_argument("--testing_file", default="diplomacy_cv.csv", type=str, help="testing file path")
    args = parser.parse_args()

    # training and validation
    if args.tf_idf:
        x, y, word_list, word2idx, tf_idf = train_preprocess(args)
    else:
        x, y, word_list, word2idx = train_preprocess(args)
    log_reg =logistic_regression(x, y)
    log_reg.train(args)
    for i in range(5):
        print(f"fold {i+1} val acc: {log_reg.fold_acc[i]}")

    # testing
    if args.tf_idf:
        x, y = test_preprocess(args, word_list, word2idx, tf_idf)
    else:
        x, y = test_preprocess(args, word_list, word2idx)
    pred = log_reg.test(x, y)