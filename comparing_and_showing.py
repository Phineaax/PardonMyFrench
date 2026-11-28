import score_utils as scoring
import chunking 

import pandas as pd
import numpy as np

from nltk.langnames import langname, langcode

from scipy.linalg import norm as normalize_l2
from scipy.spatial import distance_matrix as dm
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import squareform, pdist

import matplotlib.pyplot as plt

import random

import math



def pront(verbose, text) : 
    if verbose : 
        print(text)

SABLE  = "#F2EACF"   # fond
BRIQUE = "#BF3B24"   # accent 1 : titres, branches
ACIER  = "#5D8FA8"   # accent 2
JAUNE   = "#EE7A0B"   # accent 3 : texte courant
CREME  = "#FBF4E2"   # complément
BLANC = "#FCF8ED"

# Couleurs des personas. La charte n'a que 4 teintes, dont le crème, trop pâle
# sur fond sable. Pour plus de 3 personas, j'ai ajouté des teintes assorties
# (ocre, vert-de-gris, prune) qui ne font pas partie de la charte.
PERSONA_PALETTE = [BRIQUE, ACIER, JAUNE, "#8BC34A", "#D92E6A", "#E0A526", "#C87978"]


available_metrics = scoring.available_metrics
# Les scorers possibles sont : 
# [<scorer_longMoy>,
#  <scorer_divLex_HDD>,
#  <scorer_divLex_MTLD>,
#  <scorer_densLex>,
#  <scorer_bowVec>,
#  <scorer_n_grams_motsVec>,
#  <scorer_n_grams_charVec>]

def create_baseline(json_path, optional_df=None, true_names = False, n=-1, by_lines=False) :
    if optional_df is None : 
        df = chunking.artist_folder_to_artist_df(json_path)
    else : 
        df = optional_df.copy()

    if n == -1 : 
        n = len(list(df["persona_id"].unique()))

    if n > 26 :
        n=26
        
    if true_names : 
        fake_personas = list(df["persona_id"].unique())
    else : 
        fake_personas = [chr(65+i)*4 for i in range(n)]

    if by_lines : 
        df = df.sample(frac=1).reset_index(drop=True)
        df["persona_id"] = [fake_personas[i%n] for i in range(len(df))]

    else : 
        songs = list(df["song_id"].unique())
        
        baseline_song_to_persona = {s:random.choice(fake_personas) for i, s in enumerate(songs)}

        df["persona_id"] = df['song_id'].apply(lambda s : baseline_song_to_persona[s])
    return df.reset_index(drop=True)

def get_final_results(path, scorers=scoring.available_metrics, n=3, chunk_tokens_nb:int=chunking.DEFAULT_CHUNK_LINES_NB, vocabs_dict=dict(), optional_df = None, do_show = False, by_song=False, verbose = True) :

    """QUELQUES EXPLICATIONS : 
    Scorers est une liste des métriques que vous voulez intégrer. Les scorings différents sont dans scoring.available_metrics.
    Path est le chemin vers le DOSSIER qui contient les .json formatés comme on a dit
    Le n c'est pour les n-grams, mais ça sert à rien si vous vous en servez pas
    """
    if optional_df is None : 
        lyrics_df = chunking.artist_folder_to_artist_df(path).reset_index()
    else : 
        lyrics_df = optional_df
    #Pour les stop words, le langage le plus fréquent est considéré comme le langage global. On transforme le code de langue en nom de langue
    language = langname(lyrics_df["language"].value_counts().index[np.argmax(lyrics_df["language"].value_counts())])

    #On ne crée les vocabulaires QUE si ils sont demandés, ou si on ne les a pas déjà dans les paramètres via vocabs_dict
    do_voc = (scoring.bowVec in scorers, scoring.n_grams_motsVec in scorers, scoring.n_grams_charVec in scorers, scoring.n_grams_phonVec in scorers)

    all_lines = list(lyrics_df["text"]) if any(do_voc) else None
    
    if vocabs_dict == dict() : 
        voc_bow = scoring.create_vocab_bow(all_lines) if do_voc[0] else None
        if voc_bow : pront(verbose, "vocabulaire Bag Of Words créé")
        voc_n_mots = scoring.create_vocab_n_grams_mots(all_lines, n) if do_voc[1] else None
        if voc_n_mots : pront(verbose, "vocabulaire n-grams de mot créé")
        voc_n_char = scoring.create_vocab_n_grams_char(all_lines, n) if do_voc[2] else None
        if voc_n_char : pront(verbose, "vocabulaire n-grams de caractères créé")
        voc_n_phonem = scoring.create_vocab_n_grams_phonem(all_lines, n) if do_voc[3] else None
        if voc_n_phonem : pront(verbose, "vocabulaire n-grams de phonemes créé")
        voc_kld = scoring.create_kld_vocab(all_songs=list(chunking.lines_df_to_song_df(lyrics_df))) if (scoring.kldVec in scorers) else None
        if voc_kld : pront(verbose, "vocabulaire kld créé")

    else : 
        voc_bow = vocabs_dict["voc_bow"]
        voc_n_mots = vocabs_dict["voc_n_mots"]
        voc_n_char = vocabs_dict["voc_n_char"]
        voc_n_phonem = vocabs_dict["voc_n_phonem"]
        voc_kld = vocabs_dict["voc_kld"]


    personas = pd.unique(lyrics_df["persona_id"])
    total_chunks = dict()
    
    for p in personas :
        c_p_lines = lyrics_df[lyrics_df["persona_id"] == p] #On récupère les lines de persona p
        
        if not by_song :
            shuffled_p_lines = c_p_lines.sample(frac=1).reset_index(drop=True) #Randomise
            new_chunks = chunking.format_chunk_for_scoring(shuffled_p_lines, persona_id = p, chunk_tokens_nb=chunk_tokens_nb, lang=langcode(language,2))
            total_chunks.update(new_chunks) #On ajoute les lignes
        if by_song : 
            unique_songs = list(c_p_lines["song_title"].unique())
            total_chunks.update({f"{p}_{s_id}" : list(c_p_lines[c_p_lines["song_title"] == s_id]["text"]) for i, s_id in enumerate(unique_songs)})

    chunk_ids = list(total_chunks.keys())
    results = dict()

    for scorer in scorers :
        results[scorer.name] = {key: scorer.application(total_chunks[key], 
        language=language, 
        voc_bow = voc_bow, voc_n_mots=voc_n_mots, voc_n_char=voc_n_char, voc_n_phonem=voc_n_phonem, voc_kld = voc_kld,
        n=n, 
        )
        for key in chunk_ids}
        pront(verbose, f"resultats obtenus pour {scorer}")

    non_sem = [scorer.name for scorer in scorers if "Vec" not  in scorer.name]
    pront(verbose, non_sem)
    results['non_sem_vec'] = {key : np.array([results[ns][key] for ns in non_sem]) for key in chunk_ids}

    pre_distances = {scorer : np.array(list(results[scorer].values())) for scorer in results}
    for scorer in pre_distances : 
        if len(pre_distances[scorer].shape) == 1 : #Si c'est un vecteur en 1d
            pre_distances[scorer] = pre_distances[scorer].reshape(-1, 1) #On le reshape pour en faire une suite de vecteur verticaux.

    #Normaliser en L2 ICI si tu veux : 
    #{scorer : normalize_l2(pre_distances[scorer], axis=1, ord=2) for scorer in pre_distances}
    
    distances = {scorer : (pdist(pre_distances[scorer], metric="cosine") if "Vec" in scorer else pdist(pre_distances[scorer], metric="euclidean")) for scorer in pre_distances} #On transforme les résultats en matrices de distances

    if do_show : 
        show_labels = True
        # if by_song : 
        #     show_labels = False
        
        labels = chunk_ids
        for scorer in scorers : 
            pront(verbose, f"Affichage pour {scorer}")
            create_dendrogram(distances[scorer.name], labels, scorer.description, show_labels=show_labels, method="ward")
        create_dendrogram(distances["non_sem_vec"], labels, "Vecteurs global indices non sémantiques", show_labels=show_labels, method="ward")

    return results, distances, chunk_ids

def get_aggregated_results (path, scorers=scoring.available_metrics, n=3, chunk_tokens_nb:int=chunking.DEFAULT_CHUNK_LINES_NB, n_sample=3, optional_df=None, do_show=False, verbose=True) :

    #On crée les vocabulaires ici pour ne pas avoir à les recréer à chaque fois
    do_voc = (scoring.bowVec in scorers, scoring.n_grams_motsVec in scorers, scoring.n_grams_charVec in scorers, scoring.n_grams_phonVec in scorers)
    pront(verbose, do_voc)
    vocabs_dict = dict()

    
    if optional_df is None : 
        lyrics_df = chunking.artist_folder_to_artist_df(path).reset_index()
    
    else : 
        lyrics_df = optional_df

    if any(do_voc) :
        all_lines = list(lyrics_df["text"])
    else : 
        all_lines = []

    voc_bow = scoring.create_vocab_bow(all_lines) if do_voc[0] else None
    voc_n_mots = scoring.create_vocab_n_grams_mots(all_lines, n) if do_voc[1] else None
    vocab_n_char = scoring.create_vocab_n_grams_char(all_lines, n) if do_voc[2] else None
    voc_n_phonem = scoring.create_vocab_n_grams_phonem(all_lines, n) if do_voc[3] else None
    voc_kld = scoring.create_kld_vocab(all_songs=list(chunking.lines_df_to_song_df(lyrics_df))) if (scoring.kldVec in scorers) else None

    vocabs_dict = {
        "voc_bow" : voc_bow,
        "voc_n_mots" : voc_n_mots,
        "voc_n_char" : vocab_n_char,
        "voc_n_phonem" : voc_n_phonem,
        "voc_kld" : voc_kld
    }

    for v in vocabs_dict : 
        if vocabs_dict[v] :
            pront(verbose, f"Vocabulaire créé pour {v}")

    pront(verbose, f"------------------------------\nRécupération des résultats n°1")
    r, average_global_distances, chunk_ids = get_final_results(path=path, scorers=scorers, n=n, chunk_tokens_nb=chunk_tokens_nb, vocabs_dict=vocabs_dict, optional_df=optional_df, verbose=verbose)
    
    scorers_names = [scorer.name for scorer in scorers] + ['non_sem_vec']
    
    for i in range(n_sample-1) :
        pront(verbose, f"------------------------------\nRécupération des résultats n°{i+2}")
        _, distances, _ = get_final_results(path=path, scorers=scorers, n=n, chunk_tokens_nb=chunk_tokens_nb, vocabs_dict=vocabs_dict, optional_df=optional_df, verbose=verbose)
        #On additione
        for scorer in scorers_names :
            average_global_distances[scorer] += distances[scorer]
    
    for scorer in scorers_names :
        average_global_distances[scorer] += average_global_distances[scorer]/n_sample

    if do_show : 
        labels = chunk_ids
        for scorer in scorers_names : 
            pront(verbose, f"Affichage pour {scorer}")
            create_dendrogram(average_global_distances[scorer], labels, scorer)

    return average_global_distances, chunk_ids

## Fonctions plot générées sauvagement


def get_root(label):
    """'mon_artiste_12' -> 'mon_artiste' (on coupe au DERNIER underscore)"""
    return label.rsplit("_", 1)[0]

def make_color_map(labels, palette=PERSONA_PALETTE):
    roots = sorted(set(get_root(l) for l in labels))
    return {r: palette[i % len(palette)] for i, r in enumerate(roots)}

def create_dendrogram(distances, labels, title, ax=None, color_map=None, roots=None, show_labels=True, method="average"):
    X = distances

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor(CREME)

    Z = linkage(X, method=method)#, metric=metric)
    
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=90, leaf_font_size=8,
               link_color_func=lambda k: BRIQUE)   # branches en rouge brique

    # Fond, titre, axes
    ax.set_facecolor(CREME)
    ax.set_title(title, color=BRIQUE, fontweight="bold")
    ax.tick_params(axis="y", colors=BRIQUE)
    for spine in ax.spines.values():
        spine.set_color(BRIQUE)

    # Étiquettes colorées par persona
    if roots is None:
        roots = [get_root(l) for l in labels]
    label_to_root = dict(zip(labels, roots))
    if color_map is None:
        color_map = make_color_map(labels)

    for tick in ax.get_xmajorticklabels():
        root = label_to_root.get(tick.get_text())
        tick.set_color(color_map.get(root, BRIQUE))

    if not show_labels :
        ax.set_xticklabels(['▬▬▬' for _ in labels])

    return ax

def plot_all_dendrograms(path, scorers=scoring.available_metrics, n=3, ncols=2,
                         figsize_per_plot=(9, 5), chunk_tokens_nb:int=chunking.DEFAULT_CHUNK_LINES_NB, n_sample=3,
                         optional_df = None, show_ids = False):
    
    distances, chunk_ids = get_aggregated_results(path, scorers=scorers, n=n, chunk_tokens_nb=chunk_tokens_nb, n_sample=n_sample, optional_df=optional_df)

    nrows = math.ceil((len(scorers)+1) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                         figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
                         squeeze=False)
    fig.patch.set_facecolor(BLANC)
    axes = axes.flatten()

    color_map = make_color_map(chunk_ids)  # mêmes couleurs dans tous les sous-graphes

    scorers_names = [scorer.name for scorer in scorers] + ['non_sem_vec']
    scorer_description = [scorer.description for scorer in scorers] + ['vecteur des indices non sémantiques']


    roots = [get_root(c) for c in chunk_ids]
    color_map = make_color_map(chunk_ids)

    labels = roots if not show_ids else chunk_ids

    for ax, s_n, s_d in zip(axes, scorers_names, scorer_description):
        create_dendrogram(distances[s_n], labels, s_d,
                        ax=ax, color_map=color_map, roots=roots)


    for ax in axes[len(scorers)+1:]:  # on cache les axes en trop
        ax.axis("off")

    fig.tight_layout()
    plt.show()
    return distances