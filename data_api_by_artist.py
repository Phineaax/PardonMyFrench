#Choper un token ici : https://genius.com/api-clients

import pandas as pd
from lyricsgenius import Genius

metadatas_a_garder = ['artist_names', 'full_title', 'id', 'path', 'title', 'featured_artists', 'lyrics']

def get_all_songs(artist_name, access_token) : 
    """Récupère toutes les musiques qu'un'e artiste a créé en passant par les albums qu'iel a sorti"""
    
    genius = Genius(access_token)

    artist = genius.search_artist(artist_name, get_full_info=False, max_songs=0) #On récupère la fiche de l'artiste

    if artist == None :
        print("Nom d'artiste non reconnu")
        return None
    
    artist_id = artist.to_dict()["id"]
    albums = genius.artist_albums(artist_id)["albums"] #On récupère ses albums
    albums_id = [a["id"] for a in albums]

    results = []

    for a_id in albums_id[0:1] : 
        tracks_id = [t["song"]["id"] for t in genius.album_tracks(a_id)["tracks"]] #On itère sur les albums un par un, mais la variable "track" ne donne pas les paroles
        for t_id in tracks_id : 
            song = genius.search_song(song_id=t_id) #Pour avoir les paroles, on a besoin de retrouver les titres un par un (c'est ce qui est lent)
            d = song.to_dict() #On converti le titre en dictionnaire
            d = {p:d[p] for p in d if p in metadatas_a_garder} #On crée un sous dictionnaire en ne gardant que les clefs contenues dans la liste "à garder"
            d["album_id"] = a_id 
            d["artist_id"] = artist_id
            results.append(d)

    return pd.DataFrame(results) #On renvoie la base de données correspondantes
