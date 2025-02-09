import numpy 
import csv
import re
import random
import numpy as np


def read_file(threshold=1):

    tuples = []

    with open("snli.csv") as f:
        csv_reader = csv.reader(f, delimiter=",")
        for row in csv_reader:
            line = row[2]
            line_tokens = re.sub(r"[^a-zA-Z0-9\s]", " ", line).split()
            line_tokens = [token.lower() for token in line_tokens]

            tuples.append(line_tokens)
        
        words = [word for word_list in tuples for word in word_list]
        vocab, count = np.unique(np.array(words), return_counts=True)
        left_vocab = vocab[count>threshold]
        remove_vocab = vocab[count<=threshold]
        print(f"deleted vocab number: {remove_vocab.size}")
        print(f"left vocab number: {left_vocab.size}")
        print(f"original word number: {np.array(words).size}")
        tuples = [[word for word in word_line if word in left_vocab] for word_line in tuples]
        words = [word for word_list in tuples for word in word_list]
        print(f"filted word number: {np.array(words).size}")       

    return tuples, left_vocab


def create_term_context_matrix(line_tuples, vocab, context_window_size=1):

    vocab_dict = {word: i for i, word in enumerate(vocab)}
    n = len(vocab)
    tc_matrix = np.zeros((n, n), dtype=np.int32)

    for tokens in line_tuples:
        token_indices = [vocab_dict[w] for w in tokens if w in vocab_dict]
        
        for i, r_idx in enumerate(token_indices):
            start = max(0, i - context_window_size)
            end = min(len(token_indices), i + context_window_size + 1)  
            for j in range(start, end):
                if i != j: 
                    c_idx = token_indices[j]
                    tc_matrix[r_idx, c_idx] += 1
    
    return tc_matrix


def create_ppmi_matrix(term_context_matrix):

    total_num = np.sum(term_context_matrix)

    r_total_num = np.sum(term_context_matrix, axis=1, keepdims=True) 
    c_total_num = np.sum(term_context_matrix, axis=0, keepdims=True)
    p_ij = term_context_matrix / total_num

    p_i = r_total_num / total_num
    p_j = c_total_num / total_num

    denominator = p_i @ p_j 
    with np.errstate(divide='ignore', invalid='ignore'): 
        pmi = np.log2(np.where(denominator > 0, p_ij / denominator, 1))  
    pmi[np.isneginf(pmi) | np.isnan(pmi)] = 0
    
    ppmi = np.maximum(pmi, 0)

    return ppmi


def rank_words(target_word_index, matrix):

    target_vector = matrix[target_word_index]

    similarities = np.array([compute_cosine_similarity(target_vector, matrix[i]) for i in range(matrix.shape[0])])
    sorted_indices = np.argsort(-similarities)
    sorted_similarities = similarities[sorted_indices]

    return sorted_indices.tolist(), sorted_similarities.tolist()


def compute_cosine_similarity(vector1, vector2):
    
    vector1 = np.array(vector1, dtype=np.float64)
    vector2 = np.array(vector2, dtype=np.float64)
    
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    cosine_similarity = np.dot(vector1, vector2) / (norm1 * norm2)
    
    return cosine_similarity

if __name__ == "__main__":
    print("Loading and filtering dataset")
    tuples, vocab = read_file(threshold=3)
    vocab2idx = dict((vocab, i) for i, vocab in enumerate(vocab))

    print("Calculate term context matrix")
    tc_matrix = create_term_context_matrix(tuples, vocab, context_window_size=3)

    print("Calculate PPMI matrix")
    ppmi = create_ppmi_matrix(tc_matrix)

    word = "man"
    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on term-context frequency matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab2idx[word], tc_matrix)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))
    
    word = "man"
    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on PPMI matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab2idx[word], ppmi)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))

    word = "woman"
    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on term-context frequency matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab2idx[word], tc_matrix)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))
    
    word = "woman"
    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on PPMI matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab2idx[word], ppmi)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))

    word = "boy"
    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on term-context frequency matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab2idx[word], tc_matrix)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))

    word = "boy"
    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on PPMI matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab2idx[word], ppmi)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))
    
    word = "girl"
    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on term-document frequency matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab2idx[word], tc_matrix)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))
    
    word = "girl"
    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on PPMI matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab2idx[word], ppmi)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))
