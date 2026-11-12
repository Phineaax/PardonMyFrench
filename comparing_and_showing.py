import score_utils as scoring
import chunking 

import pandas as pd
import numpy as np

from nltk.langnames import langname

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial import distance_matrix as dm
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt


from random import shuffle
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

def create_baseline(json_path, optional_df=None, true_names = False, n=-1) :
    if not any(optional_df) : 
        df = chunking.artist_folder_to_artist_df(json_path)
    else : 
        df = optional_df.copy()
        
    if n == -1 : 
        n = len(list(df["persona_id"].unique()))

    if true_names : 
        fake_personas = list(df["persona_id"].unique())
    else : 
        fake_personas = [chr(65+i)*4 for i in range(n)]
    songs = list(df["song_id"].unique())
    shuffle(songs)
    
    baseline_song_to_persona = {s:fake_personas[i%n] for i, s in enumerate(songs)}

    df["persona_id"] = df['song_id'].apply(lambda s : baseline_song_to_persona[s])
    return df

# def create_dendrogram(distance_matrix, labels, title) : 
#     y = squareform(distance_matrix)
    
#     Z = linkage(y, method="ward")

#     plt.figure(figsize=(14, 6))
#     dendrogram(Z, labels=labels, leaf_rotation=90)
#     plt.title(f"Clustering hiérarchique selon : {title}")
#     plt.tight_layout()
#     plt.show()

def get_final_results(path, scorers=scoring.available_metrics, n=3, chunk_tokens_nb:int=1500, vocabs_dict=dict(), optional_df = [], do_show = False, by_song=False) :

    """QUELQUES EXPLICATIONS : 
    Scorers est une liste des métriques que vous voulez intégrer. Les scorings différents sont dans scoring.available_metrics.
    Path est le chemin vers le DOSSIER qui contient les .json formatés comme on a dit
    Le n c'est pour les n-grams, mais ça sert à rien si vous vous en servez pas
    """
    if not any(optional_df) : 
        lyrics_df = chunking.artist_folder_to_artist_df(path).reset_index()
    else : 
        lyrics_df = optional_df

    #Pour les stop words, le langage le plus fréquent est considéré comme le langage global. On transforme le code de langue en nom de langue
    language = langname(lyrics_df["language"].value_counts().index[np.argmax(lyrics_df["language"].value_counts())])

    #On ne crée les vocabulaires QUE si ils sont demandés, ou si on ne les a pas déjà dans les paramètres via vocabs_dict
    do_voc = (scoring.bowVec in scorers, scoring.n_grams_motsVec in scorers, scoring.n_grams_charVec in scorers, scoring.n_grams_phonVec in scorers)

    all_lines = list(lyrics_df["text"]) if any(do_voc) else None
    
    if 'voc_bow' not in vocabs_dict : 
        voc_bow = scoring.create_vocab_bow(all_lines) if do_voc[0] else None
        if voc_bow : print("vocabulaire Bag Of Words créé")
    else : voc_bow = vocabs_dict["voc_bow"]
    
    
    if 'voc_n_mots' not in vocabs_dict : 
        voc_n_mots = scoring.create_vocab_n_grams_mots(all_lines, n) if do_voc[1] else None
        if voc_n_mots : print("vocabulaire n-grams de mot créé")
    else : voc_n_mots = vocabs_dict["voc_n_mots"]
    
    
    if 'voc_n_char' not in vocabs_dict : 
        voc_n_char = scoring.create_vocab_n_grams_char(all_lines, n) if do_voc[2] else None
        if voc_n_char : print("vocabulaire n-grams de caractères créé")
    else : voc_n_char = vocabs_dict["voc_n_char"]
    
    
    if 'voc_n_phonem' not in vocabs_dict : 
        voc_n_phonem = scoring.create_vocab_n_grams_phonem(all_lines, n) if do_voc[3] else None
        if voc_n_phonem : print("vocabulaire n-grams de phonemes créé")
    else : voc_n_phonem = vocabs_dict["voc_n_phonem"]
    
    
    del all_lines
        #Fin de création des vocabulaires

    personas = pd.unique(lyrics_df["persona_id"])

    total_chunks = dict()
    
    for p in personas :
        c_p_lines = lyrics_df[lyrics_df["persona_id"] == p] #On récupère les lines de persona p
        
        if not by_song :
            shuffled_p_lines = c_p_lines.sample(frac=1).reset_index(drop=True) #Randomise
            new_chunks = chunking.format_chunk_for_scoring(shuffled_p_lines, persona_id = p, chunk_tokens_nb=chunk_tokens_nb)
            total_chunks.update(new_chunks) #On ajoute les lignes
        if by_song : 
            unique_songs = list(c_p_lines["song_title"].unique())
            total_chunks.update({f"{p}_{s_id}" : list(c_p_lines[c_p_lines["song_title"] == s_id]["text"]) for i, s_id in enumerate(unique_songs)})

    chunk_ids = list(total_chunks.keys())

    results = dict()

    for scorer in scorers :
        results[scorer.name] = {key: scorer.application(total_chunks[key], 
        language=language, voc_bow = voc_bow, voc_n_mots=voc_n_mots, voc_n_char=voc_n_char, voc_n_phonem=voc_n_phonem
            ) 
        for key in chunk_ids}
        print(f"resultats obtenus pour {scorer}")

    non_sem = [scorer.name for scorer in scorers if "Vec" not  in scorer.name]
    results['non_sem_vec'] = {key : np.array([results[ns][key] for ns in non_sem]) for key in chunk_ids}

    distances = {scorer : np.array(list(results[scorer].values())) for scorer in results}
    for scorer in distances : 
        if len(distances[scorer].shape) == 1 : #Si c'est un vecteur en 1d
            distances[scorer] = distances[scorer].reshape(-1, 1) #On le reshape pour en faire une suite de vecteur verticaux.

    distances = {scorer : dm(distances[scorer], distances[scorer]) for scorer in distances} #On transforme les résultats en matrices de distances

    if do_show : 
        labels = chunk_ids
        for scorer in scorers : 
            print(f"Affichage pour {scorer}")
            create_dendrogram(distances[scorer.name], labels, scorer.description)
        create_dendrogram(distances["non_sem_vec"], labels, "Vecteurs global indices non sémantiques")

    return results, distances, chunk_ids

def get_aggregated_results (path, scorers=scoring.available_metrics, n=3, chunk_tokens_nb:int=1500, n_sample=8, optional_df=[], do_show=False) :

    #On crée les vocabulaires ici pour ne pas avoir à les recréer à chaque fois
    do_voc = (scoring.bowVec in scorers, scoring.n_grams_motsVec in scorers, scoring.n_grams_charVec in scorers, scoring.n_grams_phonVec in scorers)
    vocabs_dict = dict()
    if any(do_voc) :
        if not any(optional_df) : 
            lyrics_df = chunking.artist_folder_to_artist_df(path).reset_index()
        else : 
            lyrics_df = optional_df
        all_lines = list(lyrics_df["text"])
    else : 
        all_lines = []
        
    voc_bow = scoring.create_vocab_bow(all_lines) if do_voc[0] else None
    voc_n_mots = scoring.create_vocab_n_grams_mots(all_lines, n) if do_voc[1] else None
    vocab_n_char = scoring.create_vocab_n_grams_char(all_lines, n) if do_voc[2] else None
    voc_n_phonem = scoring.create_vocab_n_grams_phonem(all_lines, n) if do_voc[3] else None
    vocabs_dict = {
        "voc_bow" : voc_bow,
        "voc_n_mots" : voc_n_mots,
        "voc_n_char" : vocab_n_char,
        "voc_n_phonem" : voc_n_phonem
    }
    for v in vocabs_dict : 
        if vocabs_dict[v] :
            print(f"Vocabulaire créé pour {v}")

    print(f"------------------------------\nRécupération des résultats n°1")
    r, average_global_distances, chunk_ids = get_final_results(path=path, scorers=scorers, n=n, chunk_tokens_nb=chunk_tokens_nb, vocabs_dict=vocabs_dict, optional_df=optional_df)
    
    scorers_names = [scorer.name for scorer in scorers] + ['non_sem_vec']
    
    for i in range(n_sample-1) :
        print(f"------------------------------\nRécupération des résultats n°{i+2}")
        _, distances, _ = get_final_results(path=path, scorers=scorers, n=n, chunk_tokens_nb=chunk_tokens_nb, vocabs_dict=vocabs_dict, optional_df=optional_df)
        #On additione
        for scorer in scorers_names :
            average_global_distances[scorer] += distances[scorer]
    
    for scorer in scorers_names :
        average_global_distances[scorer] += average_global_distances[scorer]/n_sample

    if do_show : 
        labels = chunk_ids
        for scorer in scorers_names : 
            print(f"Affichage pour {scorer}")
            create_dendrogram(average_global_distances[scorer], labels, scorer)

    return average_global_distances, chunk_ids

## Fonctions plot générées sauvagement


def get_root(label):
    """'mon_artiste_12' -> 'mon_artiste' (on coupe au DERNIER underscore)"""
    return label.rsplit("_", 1)[0]

def make_color_map(labels, cmap_name="tab20"):
    roots = sorted(set(get_root(l) for l in labels))
    cmap = plt.get_cmap(cmap_name)
    return {r: cmap(i % cmap.N) for i, r in enumerate(roots)}

def create_dendrogram(distances, labels, title, ax=None, color_map=None):

    #labels = [l[:l.rfind("_")] for l in labels]
    
    y = squareform(distances)
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    # result_dict: {chunk_id: vecteur ou scalaire} -> matrice (n_chunks, n_features)
    
    Z = linkage(y, method="ward")  # ou ta méthode habituelle
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=90, leaf_font_size=8)
    ax.set_title(title)

    # Coloration des labels par racine
    if color_map is None:
        color_map = make_color_map(labels)
    for tick in ax.get_xmajorticklabels():
        tick.set_color(color_map[get_root(tick.get_text())])

    return ax

def plot_all_dendrograms(path, scorers=scoring.available_metrics, n=3, ncols=2,
                         figsize_per_plot=(9, 5), chunk_tokens_nb:int=1500, n_sample=8,
                         optional_df = []):
    distances, chunk_ids = get_aggregated_results(path, scorers=scorers, n=n, chunk_tokens_nb=chunk_tokens_nb, n_sample=n_sample, optional_df=optional_df)

    nrows = math.ceil(len(scorers) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
                             squeeze=False)
    axes = axes.flatten()

    color_map = make_color_map(chunk_ids)  # mêmes couleurs dans tous les sous-graphes

    for ax, scorer in zip(axes, scorers):
        create_dendrogram(distances[scorer.name], chunk_ids, scorer.description, 
                          ax=ax, color_map=color_map)

    for ax in axes[len(scorers):]:  # on cache les axes en trop
        ax.axis("off")

    fig.tight_layout()
    plt.show()
    return distances