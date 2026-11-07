import score_utils as scoring
import chunking 
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

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
    do_voc = (scoring.bowVec in scorers, scoring.n_grams_motsVec in scorers, scoring.n_grams_charVec in scorers)

    all_lines = list(lyrics_df["text"]) if any(do_voc) else None
    voc_bow = scoring.create_vocab_bow(all_lines) if do_voc[0] else None
    print("vocabulaire Bag Of Words créé")

        
    voc_n_mots = scoring.create_vocab_n_grams_mots(all_lines, n) if do_voc[1] else None
    if voc_n_mots : print("vocabulaire n-grams de mot créé")
    voc_n_char = scoring.create_vocab_n_grams_char(all_lines, n) if do_voc[2] else None
    if voc_n_char : print("vocabulaire n-grams de caractères créé")
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
        results[scorer.name] = {key: scorer.application(total_chunks[key], language="english", voc_bow = voc_bow, voc_n_mots=voc_n_mots, voc_n_char=voc_n_char) for key in chunk_ids}
        print(f"resultats obtenus pour {scorer}")

    labels = chunk_ids
    for scorer in scorers : 
        print(f"Affichage pour {scorer}")
        create_dendrogram(results[scorer.name], labels, scorer.description)

    return total_chunks