import numpy as np
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from lexicalrichness import LexicalRichness as LR

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')

from collections import Counter

from nltk.langnames import langname, langcode
import pyphen

from kld_bow import DispersionBoW
#from sentence_transformers import SentenceTransformer

word_tokenizer = word_tokenize

universal_pos_tagset = ["VERB", "NOUN", "PRON", "ADJ", "ADV", "ADP", "CONJ", "DET", "NUM", "PRT", "X", "."]
vocab_pos2i = TfidfVectorizer().fit(universal_pos_tagset)

def voc_n_pos(n:int) :
    u = [p.lower() for p in universal_pos_tagset]
    total = [p for p in u] 
    for _ in range(n-1) :
        for _ in range(len(total)) : 
            x = total.pop(0)
            total.extend([x+" "+p for p in u])

    vectorizer = create_vocab_n_grams_mots(total, n)
    return vectorizer

def pos_tag_chunk(lines_chunk:List[str]) -> List[str] :
    return [
        " ".join([tag[1] for tag in nltk.pos_tag(word_tokenizer(l), tagset="universal") 
        ]) 
     for l in lines_chunk]

def wordify(lines_chunk:List[str]) -> int:
    """Transforme une liste de phrase en liste de mots"""
    words = [word_tokenizer(line.lower()) for line in lines_chunk]
    return sum(words, start=[])

class Mesure() :
    def __init__(self, name, application, description) :
        self.name = name
        self.application = application
        self.description = description

    def __repr__(self) : 
        return f"<scorer_{self.name}>"

def longueur_moyenne(lines_chunk:List[str], **kwargs) -> float:
    words = wordify(lines_chunk)
    words_length = list(map(lambda w : len(w), words))
    
    return sum(words_length)/len(words_length)
longMoy = Mesure("longMoy", longueur_moyenne, "Words mean length")

def syllabe_moyenne(lines_chunk:List[str], language, **kwargs) -> float:
    lang = langcode(language, 2)
    if lang == "en" : lang+="-us"
    dic = pyphen.Pyphen(lang=lang)

    hyphen = "#$µ"

    syll_per_line = list(map(lambda l : dic.inserted(l, hyphen=hyphen).count(hyphen)+1, lines_chunk)) #On compte les syllabes
    
    return sum(syll_per_line)/len(lines_chunk)
syllMoy = Mesure("syllMoy", syllabe_moyenne, "Average syllabes count per line")

def diversite_lexicale_HDD(lines_chunk:List[str], **kwargs) -> float:
    words = wordify(lines_chunk)
    n_draws =  42
    if len(words)<n_draws :
        n_draws = len(words)//2
    return LR(words, preprocessor=None, tokenizer=None).hdd(draws=n_draws)
divLex_HDD = Mesure("divLex_HDD", diversite_lexicale_HDD,  "Lexical diversity score (HDD)")


def diversite_lexicale_MTLD(lines_chunk, **kwargs) :
    words = wordify(lines_chunk)
    return LR(words, preprocessor=None, tokenizer=None).mtld()
divLex_MTLD = Mesure("divLex_MTLD", diversite_lexicale_MTLD, "Lexical diversity score (MTLD)")


def densite_lexicale(lines_chunk:List[str], language, **kwargs) -> float:
    stop_words = set(stopwords.words(language))
    words = wordify(lines_chunk)
    full_words = [w for w in words if w not in stop_words]
    return len(full_words)/len(words)
densLex = Mesure("densLex", densite_lexicale, "Lexical density")

def formatted_bow(lines, vec) : 

    x = sum(vec.transform(lines))
    u = np.reshape(x.toarray(), x.shape)
    v = normalize(u[0][:,np.newaxis], axis=0).ravel()
    return v

#BOW
def create_vocab_bow(all_lines) -> Dict[str, int]:
    """Crée le vocab total pour faire des Bag Of Words indépendamment. Prend en argument la grande liste de toutes les lignes."""
    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_lines)

    return vectorizer

def bow(lines_chunk:List[str], voc_bow, **kwargs) -> np.array:
    """Renvoie un vecteur Bag Of Words"""

    return formatted_bow(lines_chunk, voc_bow)
bowVec = Mesure("bowVec", bow, "BOW")
posVec = Mesure("posVec", lambda lc, voc_bow=None, **k : bow(lines_chunk=pos_tag_chunk(lc), voc_bow=vocab_pos2i, **k), "BOW of POS tags")


#N-grams mots
def create_vocab_n_grams_mots(all_lines, n) -> Dict[str, int]:
    """Crée le vocabulaire pour utiliser la fonction n_gram_mots. Prend en argument la grande liste de toutes les lignes."""
    vectorizer = TfidfVectorizer(ngram_range=(n, n))
    vectorizer.fit(all_lines)

    return vectorizer

def n_grams_mots(lines_chunk:List[str], voc_n_mots, **kwargs) -> np.array:

    return formatted_bow(lines_chunk, voc_n_mots)

n_grams_motsVec = Mesure("n_grams_motsVec", n_grams_mots, "BOW of words n-grams")
n_grams_posVec = Mesure("n_grams_posVec", lambda lc, voc_n_mots=None, **k : n_grams_mots(lines_chunk=pos_tag_chunk(lc), voc_n_mots=voc_n_pos(k['n']), **k), "BOW of POS tags n-grams")

#N-grams char
def create_vocab_n_grams_char(all_lines, n) -> Dict[str, int]: 
    """Crée le vocabulaire pour utiliser la fonction n_grams_char. Prend en argument la grande liste de toutes les lignes."""
    vectorizer = TfidfVectorizer(ngram_range=(n, n))
    vectorizer.fit(all_lines)

    return vectorizer

def create_vocab_n_grams_phonem(all_lines, n) -> Dict[str, int]: 
    """Phonétise puis crée le vocabulaire de phonemes pour utiliser la fonction n_grams_char. Prend en argument la grande liste de toutes les lignes."""

    #Phonétisation
    phonetize = lambda x : x
    all_lines = list(map(phonetize, all_lines))#Remplacer f par une fonction de phonétisation str->str

    return create_vocab_n_grams_char(all_lines, n) 

def n_grams_char(lines_chunk:List[str], voc_n_char=None, voc_n_phonem=None, **kwargs) -> np.array:

    if voc_n_char :
        vecab_w2i = voc_n_char
    elif voc_n_phonem :
        vecab_w2i = voc_n_phonem

    return formatted_bow(lines_chunk, vecab_w2i)
n_grams_charVec = Mesure("n_grams_charVec", n_grams_char, "BOW of characters n-grams")
n_grams_phonVec = Mesure("n_grams_phonVec", n_grams_char, "BOW of phonemes n-grams")

def create_kld_vocab(all_songs) :
    vectorizer = DispersionBoW()
    vectorizer.fit(all_songs)

    return vectorizer

def kld_bow(lines_chunk:List[str], voc_kld, **kwargs) -> np.array:

    return formatted_bow(lines_chunk, voc_kld)
kldVec = Mesure("kldVec", kld_bow, "BOW with kld dispersion ponderation")

# def chunk_to_embedding(lines_chunk:List[str]) :
#     model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
#     vector_sentences = model.encode(lines_chunk)
#     vectorized = np.mean(vector_sentences, axis=0)
#     return vectorized
# sentenceTransformerVec = Mesure("chunk_to_embedding", chunk_to_embedding)

all_metrics = [longMoy, divLex_MTLD, densLex, syllMoy, divLex_HDD, n_grams_posVec, posVec, bowVec, n_grams_motsVec, n_grams_charVec, n_grams_phonVec]
available_metrics = [longMoy, divLex_HDD, densLex, syllMoy, n_grams_posVec, posVec, bowVec, n_grams_motsVec, n_grams_charVec, kldVec]#, divLex_MTLD, n_grams_phonVec]
non_vec_metrics = [longMoy, divLex_HDD, densLex, syllMoy,]
vec_metrics = [n_grams_posVec, posVec, bowVec, n_grams_motsVec, n_grams_charVec, kldVec]