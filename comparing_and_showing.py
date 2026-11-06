import numpy as np
import chunking
import score_utils

#EBAUCHE DE CODE POUR LES COMPARAISONS ET VISUALISATION

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def data_from_persona(persona_chunks, scorer) :
    data_output = []
    for c in persona_chunks :
        data_output.append(scorer(c))

    return data_output

def dendrogram(X, method='ward', distance='cosine') : 
    #Code trouvé sur "https://fr.moonbooks.org/Articles/Les-dendrogrammes-avec-Matplotlib/"
    # Effectuer le regroupement hiérarchique
    Z = linkage(X, method=method, distance=distance)
    # Tracer le dendrogramme
    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title("Dendrogramme")
    plt.xlabel("Points de données")
    plt.ylabel("Distances euclidiennes")
    plt.show()
