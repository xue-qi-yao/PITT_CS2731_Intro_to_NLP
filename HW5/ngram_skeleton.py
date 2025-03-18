import math, random

################################################################################
# Utility Functions
################################################################################

def start_pad(c):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * c

def ngrams(c, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    ngram_list = []
    for i, _ in enumerate(text):
        context = start_pad(c)
        start_idx = max(0, i-c)
        end_idx = i
        context_len = end_idx - start_idx
        if context_len:
            context = context[:(c-context_len)] + text[start_idx:end_idx]
        ngram_list.append((context, text[end_idx]))
    return ngram_list

def create_ngram_model(model_class, path, c=2):
    ''' Creates and returns a new n-gram model '''
    model = model_class(c)
    with open(path, encoding='utf-8', errors='ignore') as f:
        line = f.read()
        model.update(line)
    return model

################################################################################
# Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model without smoothing '''

    def __init__(self, c):
        self.context_len = c
        self.vocab = set()
        self.ngram = dict()

    def get_vocab(self):
        print(self.vocab)

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        ngrams_result = ngrams(self.context_len, text)
        context_list, character_list = zip(*ngrams_result)
        self.vocab = self.vocab.union(set(character_list))
        for i, context in enumerate(context_list):
            if context not in self.ngram:
                self.ngram[context] = dict()
            if character_list[i] not in self.ngram[context]:
                self.ngram[context][character_list[i]] = 1
            else:
                self.ngram[context][character_list[i]] += 1

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        if context not in self.ngram:
            return 1 / len(self.vocab)
        else:
            total_character_num = sum(self.ngram[context].values())
            if char not in self.ngram[context]:
                return 1 / total_character_num
            else:
                return self.ngram[context][char] / total_character_num

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        assert len(context) == self.context_len
        context = start_pad(self.context_len - len(context)) + context
        if context not in self.ngram:
            return random.sample(list(self.vocab), k=1)[0]
        else:
            char_list, char_count = zip(*self.ngram[context].items())
            return random.sample(char_list, k=1, counts=char_count)[0]

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        pad_text = start_pad(self.context_len)
        result = str()
        for i in range(length):
            if len(result) < self.context_len:
                if not result:
                    context = pad_text
                else:
                    context = pad_text[:self.context_len-len(result)] + result
            else:
                context = result[-self.context_len:]
            result += self.random_char(context)
        return result

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model
        Acknowledgment: 
          https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3 
          https://courses.cs.washington.edu/courses/csep517/18au/
          ChatGPT with GPT-3.5'''
        
        text = ''.join([c for c in text if c in self.vocab])
        N = len(text)
        
        char_probs = []
        for i in range(self.context_len, N):
            context = text[i-self.context_len:i]
            char = text[i]
            char_probs.append(math.log2(self.prob(context, char)))
        ppl = 2 ** (-1 * sum(char_probs) / (N - self.context_len + 2))

        return ppl


if __name__ == "__main__":
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7)
    print(m.random_text(250))
    with open("test_data/nytimes_article.txt", "r", encoding='utf-8', errors='ignore') as f:
        text = f.read()
    print(m.perplexity(text))
    with open("test_data/shakespeare_sonnets.txt", "r", encoding='utf-8', errors='ignore') as f:
        text = f.read()
    print(m.perplexity(text))

