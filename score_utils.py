import numpy as np

from lexical_diversity import lex_div as ld

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')

from collections import Counter

from sentence_transformers import SentenceTransformer

def wordify(lines_chunk) :
    words = [word_tokenize(l) for l in lines_chunk]
    return sum(words, start=[])

def create_vocab_words(all_lines) :
    words = wordify(all_lines)
    
    vocab_w2i = dict()
    i_last = 0

    for w in words : 
        if w not in vocab_w2i :
            vocab_w2i[w] = i_last
            i_last += 1

    return vocab_w2i

def create_vocab_words_n_grams(list_of_chunks, n) :
    
    vocab_w2i = dict()
    i_last = 0

    for line in list_of_chunks :
        words = word_tokenize(line)
        for i in range(len(words)-n-1) :
            t = tuple(words[i:i+n])
            if t not in vocab_w2i :
                vocab_w2i[t] = i_last
                i_last += 1

    return vocab_w2i

def create_vocab_char_n_grams(list_of_chunks, n): 
    vocab_w2i = dict()
    i_last = 0

    for line in list_of_chunks :

        for i in range(len(line)-n-1) :
            t = line[i:i+n]

            if " " not in t[1:n-1] :            
                t = tuple(t)
                
                if t not in vocab_w2i :
                    vocab_w2i[t] = i_last
                    i_last += 1

    return vocab_w2i

def longueur_moyenne(lines_chunk) :
    words = wordify(lines_chunk)
    words_length = list(map(lambda w : len(w), words))
    
    return words_length/len(words_length)

def diversite_lexicale_HDD(lines_chunk) :
    words = wordify(lines_chunk)
    return ld.hdd(words)

def diversite_lexicale_MTLD(lines_chunk) :
    words = wordify(lines_chunk)
    return ld.mtld(words)

def densite_lexicale(lines_chunk, language) :
    stop_words = set(stopwords.words(language))
    words = wordify(lines_chunk)
    words = [w for w in words if w not in stopwords]

def bow(lines_chunk, vocab_w2i) :

    v = np.zeros(len(vocab_w2i))

    words = wordify(lines_chunk)
    c = Counter(words)
    for w in c :
        v[vocab_w2i[w]] = c[w]/len(words) #BOW : le mot numéro i est noté selon le nombre de fois de son apparition  

    return v

def n_grams_mots(lines_chunk, vocab_w2i) :
    n = len(list(vocab_w2i.values)[0])
    n_grams = []
    v = np.zeros(len(vocab_w2i))

    for line in lines_chunk :
        words = word_tokenize(line)

        for i in range(len(words)-n-1) :
            t = tuple(words[i:i+n])
            n_grams.append(t)
            
    c = Counter(n_grams)
    for n_gr in c :
        v[vocab_w2i[n_gr]] = c[n_gr]/len(n_grams) #BOW : le mot numéro i est noté selon le nombre de fois de son apparition  
    return v

def n_grams_caracteres(lines_chunk, vocab_w2i) :
    n = len(list(vocab_w2i.values)[0])
    n_grams = []
    v = np.zeros(len(vocab_w2i))

    for line in lines_chunk :
        for i in range(len(line)-n-1) :
            t = line[i:i+n]

            if " " not in t[1:n-1] :            
                t = tuple(t)
                n_grams.append(t)
            
    c = Counter(n_grams)
    for n_gr in c :
        v[vocab_w2i[n_gr]] = c[n_gr]/len(n_grams) #BOW : le mot numéro i est noté selon le nombre de fois de son apparition  
    return v

def sentence_transformer_lines(lines_chunk) :
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    vector_sentences = model.encode(lines_chunk)
    vectorized = np.mean(vector_sentences, axis=0)
    return vectorized