import numpy as np

from lexicalrichness import LexicalRichness as LR

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')

from collections import Counter

#from sentence_transformers import SentenceTransformer

def wordify(lines_chunk) :
    """Transforme une liste de phrase en liste de mots"""
    words = [word_tokenize(line) for line in lines_chunk]
    return sum(words, start=[])


def longueur_moyenne(lines_chunk) :
    words = wordify(lines_chunk)
    words_length = list(map(lambda w : len(w), words))
    
    return words_length/len(words_length)

def diversite_lexicale_HDD(lines_chunk) :
    words = wordify(lines_chunk)
    return LR(words, preprocessor=None, tokenizer=None).hdd()

def diversite_lexicale_MTLD(lines_chunk) :
    words = wordify(lines_chunk)
    return LR(words, preprocessor=None, tokenizer=None).mtld()

def densite_lexicale(lines_chunk, language) :
    stop_words = set(stopwords.words(language))
    words = wordify(lines_chunk)
    words = [w for w in words if w not in stopwords]


#BOW
def create_vocab_bow(all_lines) :
    """Crée le vocab total pour faire des Bag Of Words indépendamment. Prend en argument la grande liste de toutes les lignes."""
    words = wordify(all_lines)
    
    vocab_w2i = dict()
    i_last = 0

    for w in words : 
        if w not in vocab_w2i :
            vocab_w2i[w] = i_last
            i_last += 1

    return vocab_w2i

def bow(lines_chunk, vocab_w2i) :
    """Renvoie un vecteur Bag Of Words"""
    v = np.zeros(len(vocab_w2i))

    words = wordify(lines_chunk)
    c = Counter(words)
    for w in c :
        v[vocab_w2i[w]] = c[w]/len(words) #BOW : le mot numéro i est noté selon le nombre de fois de son apparition  

    return v


#N-grams mots
def create_vocab_n_grams_mots(all_lines, n) :
    """Crée le vocabulaire pour utiliser la fonction n_gram_mots. Prend en argument la grande liste de toutes les lignes."""
    vocab_w2i = dict()
    i_last = 0

    for line in all_lines :
        words = word_tokenize(line)
        for i in range(len(words)-n-1) :
            t = tuple(words[i:i+n])
            if t not in vocab_w2i :
                vocab_w2i[t] = i_last
                i_last += 1

    return vocab_w2i

def n_grams_mots(lines_chunk, vocab_w2i) :
    n = len(list(vocab_w2i.values)[0]) #On déduit la taille de n depuis le vocab
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


#N-grams char
def create_vocab_n_grams_char(all_lines, n): 
    """Crée le vocabulaire pour utiliser la fonction n_grams_char. Prend en argument la grande liste de toutes les lignes."""

    vocab_w2i = dict()
    i_last = 0

    for line in all_lines :

        for i in range(len(line)-n-1) :
            t = line[i:i+n]

            if " " not in t[1:n-1] :            
                t = tuple(t)
                
                if t not in vocab_w2i :
                    vocab_w2i[t] = i_last
                    i_last += 1

    return vocab_w2i

def n_grams_char(lines_chunk, vocab_w2i) :
    n = len(list(vocab_w2i.values)[0]) #On déduit la taille de n depuis le vocab
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


# def chunk_to_embedding(lines_chunk) :
#     model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
#     vector_sentences = model.encode(lines_chunk)
#     vectorized = np.mean(vector_sentences, axis=0)
#     return vectorized