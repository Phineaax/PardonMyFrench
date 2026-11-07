import score_utils as scoring
import chunking 
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import math


available_metrics = scoring.available_metrics
# Les scorers possibles sont : 
# [<scorer_longMoy>,
#  <scorer_divLex_HDD>,
#  <scorer_divLex_MTLD>,
#  <scorer_densLex>,
#  <scorer_bowVec>,
#  <scorer_n_grams_motsVec>,
#  <scorer_n_grams_charVec>]

def create_dendrogram(result_dict, labels, title) : 
    X = np.array(list(result_dict.values()))

    if type(X[0]) == np.ndarray : #Si c'est un
        d = X.shape[1]    
    
    else : 
        d=1

    X = X.reshape(-1, d)

    Z = linkage(X, method="ward")

    plt.figure(figsize=(14, 6))
    dendrogram(Z, labels=labels, leaf_rotation=90)
    plt.title(f"Clustering hiérarchique selon : {title}")
    plt.tight_layout()
    plt.show()

def get_final_results(path, scorers=scoring.available_metrics, n=3) :

    """QUELQUES EXPLICATIONS : 
    Scorers est une liste des métriques que vous voulez intégrer. Les scorings différents sont dans scoring.available_metrics.
    Path est le chemin vers le DOSSIER qui contient les .json formatés comme on a dit
    Le n c'est pour les n-grams, mais ça sert à rien si vous vous en servez pas
    """

    lyrics_df = chunking.artist_folder_to_artist_df(path).reset_index()

    #On ne crée les vocabulaires QUE si ils sont demandés
    do_voc = (scoring.bowVec in scorers, scoring.n_grams_motsVec in scorers, scoring.n_grams_charVec in scorers, scoring.n_grams_phonVec in scorers)

    all_lines = list(lyrics_df["text"]) if any(do_voc) else None
    
    voc_bow = scoring.create_vocab_bow(all_lines) if do_voc[0] else None
    if voc_bow : print("vocabulaire Bag Of Words créé")
    
    voc_n_mots = scoring.create_vocab_n_grams_mots(all_lines, n) if do_voc[1] else None
    if voc_n_mots : print("vocabulaire n-grams de mot créé")
    
    voc_n_char = scoring.create_vocab_n_grams_char(all_lines, n) if do_voc[2] else None
    if voc_n_char : print("vocabulaire n-grams de caractères créé")
    
    voc_n_phonem = scoring.create_vocab_n_grams_phonem(all_lines, n) if do_voc[2] else None
    if voc_n_phonem : print("vocabulaire n-grams de phonemes créé")
    
    del all_lines
        #Fin de création des vocabulaires

    personas = pd.unique(lyrics_df["persona_id"])
    print(personas)

    p_lines = dict()
    total_chunks = dict()

    for p in personas :
        p_lines[p] = lyrics_df[lyrics_df["persona_id"] == p] #On récupère les lines de persona p
        shuffled_p_lines = p_lines[p].sample(frac=1).reset_index(drop=True) #Randomise
        total_chunks.update(chunking.format_chunk_for_scoring(shuffled_p_lines, persona_id = p)) #On ajoute les lignes

    chunk_ids = list(total_chunks.keys())

    results = dict()

    for scorer in scorers :
        results[scorer.name] = {key: scorer.application(total_chunks[key], 
        language="english", voc_bow = voc_bow, voc_n_mots=voc_n_mots, voc_n_char=voc_n_char, voc_n_phonem=voc_n_phonem
            ) 
        for key in chunk_ids}
        print(f"resultats obtenus pour {scorer}")

    labels = chunk_ids
    for scorer in scorers : 
        print(f"Affichage pour {scorer}")
        create_dendrogram(results[scorer.name], labels, scorer.description)

    return total_chunks, results, chunk_ids


## Fonctions plot générées sauvagement


def get_root(label):
    """'mon_artiste_12' -> 'mon_artiste' (on coupe au DERNIER underscore)"""
    return label.rsplit("_", 1)[0]

def make_color_map(labels, cmap_name="tab20"):
    roots = sorted(set(get_root(l) for l in labels))
    cmap = plt.get_cmap(cmap_name)
    return {r: cmap(i % cmap.N) for i, r in enumerate(roots)}

def create_dendrogram(result_dict, labels, title, ax=None, color_map=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    # result_dict: {chunk_id: vecteur ou scalaire} -> matrice (n_chunks, n_features)
    X = np.array([result_dict[l] for l in labels], dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    Z = linkage(X, method="ward")  # ou ta méthode habituelle
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=90, leaf_font_size=8)
    ax.set_title(title)

    # Coloration des labels par racine
    if color_map is None:
        color_map = make_color_map(labels)
    for tick in ax.get_xmajorticklabels():
        tick.set_color(color_map[get_root(tick.get_text())])

    return ax

def plot_all_dendrograms(path, scorers=scoring.available_metrics, n=3, ncols=2,
                         figsize_per_plot=(9, 5)):
    total_chunks, results, chunk_ids = get_final_results(path, scorers=scorers, n=n)

    nrows = math.ceil(len(scorers) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
                             squeeze=False)
    axes = axes.flatten()

    color_map = make_color_map(chunk_ids)  # mêmes couleurs dans tous les sous-graphes

    for ax, scorer in zip(axes, scorers):
        create_dendrogram(results[scorer.name], chunk_ids, scorer.description,
                          ax=ax, color_map=color_map)

    for ax in axes[len(scorers):]:  # on cache les axes en trop
        ax.axis("off")

    fig.tight_layout()
    plt.show()
    return total_chunks, results