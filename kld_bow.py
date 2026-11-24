import re
from collections import Counter, defaultdict
from math import log2
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

class DispersionBoW:
    def __init__(self, tokenize=None, sublinear=False, norm=None):
        """
        tokenize  : fonction str -> liste de tokens (défaut : mots en minuscules)
        sublinear : True -> log(1 + compte) au lieu du compte brut
        norm      : None (défaut) ou "l1" / "l2", comme le paramètre `norm` de TfidfVectorizer
        """
        self.tokenize = tokenize or (lambda s: re.findall(r"\w+", s.lower()))
        self.sublinear = sublinear
        self.norm = norm

    def fit(self, songs):
        """songs : liste de chaînes (une par chanson)."""
        toks = [self.tokenize(s) for s in songs]
        sizes = np.array([len(t) for t in toks], dtype=float)
        q = sizes / sizes.sum()

        occ = defaultdict(list)
        for i, t in enumerate(toks):
            for w, c in Counter(t).items():
                occ[w].append((i, c))

        self.weights = {}
        for w, pairs in occ.items():
            total = sum(c for _, c in pairs)
            kld = sum((c / total) * log2((c / total) / q[i]) for i, c in pairs)
            self.weights[w] = 2.0 ** -kld

        vocab = sorted(self.weights)
        self.vocabulary_ = {w: j for j, w in enumerate(vocab)}   # mot -> colonne
        self.vocab_ = vocab
        return self

    def get_feature_names_out(self):
        return np.array(self.vocab_)

    def get_weights(self, text):
        """Renvoie un dict {mot: compte pondéré} pour une chaîne."""
        out = {}
        for w, c in Counter(self.tokenize(text)).items():
            wt = self.weights.get(w, 0.0)
            if wt:
                out[w] = (np.log1p(c) if self.sublinear else c) * wt
        return out

    def transform(self, lines):
        """lines : liste de chaînes. Renvoie une csr_matrix (len(lines), len(vocabulaire))."""
        indptr, indices, data = [0], [], []
        for line in lines:
            for w, v in self.get_weights(line).items():
                j = self.vocabulary_.get(w)
                if j is not None:                    # les mots hors vocabulaire sont ignorés
                    indices.append(j)
                    data.append(v)
            indptr.append(len(indices))
        X = csr_matrix((data, indices, indptr),
                       shape=(len(lines), len(self.vocabulary_)), dtype=np.float64)
        return normalize(X, norm=self.norm) if self.norm else X