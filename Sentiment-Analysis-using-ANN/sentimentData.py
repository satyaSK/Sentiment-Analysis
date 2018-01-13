import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

positive_file_path = 'Data/positive.txt'
negative_file_path = 'Data/negative.txt'
myStemmer = WordNetLemmatizer()
n_lines = 10000000

def create_vocab(pos_file,neg_file):
    all_tokens = [] 
    for f in [pos_file, neg_file]:
        with open(f,'r') as f:
            sentences = f.readlines()
            for s in sentences[:n_lines]:
                words_in_sentence = word_tokenize(s.lower())
                all_tokens += list(words_in_sentence)
    all_tokens = [myStemmer.lemmatize(i) for i in all_tokens]
    unique_words = Counter(all_tokens)
    vocab = []
    for w in unique_words:
        if 1000 > unique_words[w] > 50:
            vocab.append(w)
    print("The size of the vocab is:", len(vocab))
    return vocab

def create_featureset(single_sample, vocab, sentiment):
    featureset = []
    with open(single_sample,'r') as f:
        sentences = f.readlines()
        for sentence in sentences[:n_lines]:
            words = word_tokenize(sentence.lower())
            words = [myStemmer.lemmatize(i) for i in words]
            features = np.zeros(len(vocab))
            
            for word in words:
                if word.lower() in vocab:
                    idx = vocab.index(word.lower())
                    features[idx] += 1
            features = list(features)
            featureset.append([features,sentiment])
    return featureset
            
def get_data(positive, negative, test_size=0.1):
    vocab = create_vocab(positive, negative)
    data = []
    data += create_featureset(positive, vocab, [1,0])
    data += create_featureset(negative, vocab, [0,1])
    random.shuffle(data)
    data = np.array(data)
    testing_size = int(test_size*len(data))
    x_train = list(data[:,0][:-testing_size])
    y_train = list(data[:,1][:-testing_size])
    x_test = list(data[:,0][-testing_size:])
    y_test = list(data[:,1][-testing_size:])
    return x_train, y_train, x_test, y_test

if __name__ =='__main__':
	x_train, y_train, x_test, y_test = get_data(positive_file_path, negative_file_path,test_size=0.1)
	with open('sentiment_data.pickle','wb') as f:
		pickle.dump([x_train, y_train, x_test, y_test], f)









