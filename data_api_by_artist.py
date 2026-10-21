#Choper un token ici : https://genius.com/api-clients

import pandas as pd
from lyricsgenius import Genius

metadatas_a_garder_albums = ['_type', 'full_title', 'title', 'id', 'path', 'release_date_components', 'primary_artists']

def get_all_albums(artist_name) : 
    """Renvoie toutes les infos nécessaires sur les albums à partir du nom de l'artiste."""
    genius = Genius(access_token)

    artist = genius.search_artist(artist_name, get_full_info=False, max_songs=0) #On récupère la fiche de l'artiste pour avoir son id

    if artist == None :
        print("Nom d'artiste non reconnu")
        return None
    
    artist_id = artist.to_dict()["id"]
    albums = genius.artist_albums(artist_id)["albums"]

    results_albums = []

    for a in albums : 
        a["tracks_id"] = [t["song"]["id"] for t in genius.album_tracks(a["id"])["tracks"]]
        a = {p:a[p] for p in a if p in metadatas_a_garder_albums}
        a["primary_artists_id"] = [ar["id"] for ar in a["primary_artists"]]
        a["primary_artists_names"] = [ar["name"] for ar in a["primary_artists"]]
        del a["primary_artists"]
        results_albums.append(a)

    return pd.DataFrame(results_albums) #On renvoie la base de données correspondante


metadatas_a_garder_tracks = ['primary_artists', 'full_title', 'title', 'id', 'path', 'featured_artists', 'writer_artists', 'lyrics', 'language']

def get_all_songs_albums(artist_name, k) : 
    """Renvoie tous les sons sortis par l'artiste, en y allant album par album."
    genius = Genius(access_token)

    artist = genius.search_artist(artist_name, get_full_info=False, max_songs=0) #On récupère la fiche de l'artiste

    if artist == None :
        print("Nom d'artiste non reconnu")
        return None
    
    artist_id = artist.to_dict()["id"]
    albums = genius.artist_albums(artist_id)["albums"]
    albums_id = [a["id"] for a in albums]

    results_tracks = []
    for i_album, a_id in enumerate(albums_id[0:k]) : 
        
        tracks_id = [t["song"]["id"] for t in genius.album_tracks(a_id)["tracks"]]

        print(tracks_id)
        for i_track, t_id in enumerate(tracks_id) : 
            print(t_id)

            try :  #Si le titre n'est pas transcrit, on a une erreur
                song = genius.search_song(song_id=t_id)
            except AssertionError as ae :
                print(ae, f"Erreur concernant le titre n°{i_track} (en comptant 0), ayant pour id {t_id}") 

            if song : #Il arrive que le song soit None, c'est entièrement la faute du référencement de l'API
                d = song.to_dict() #On converti le titre en dictionnaire
                d = {p:d[p] for p in d if p in metadatas_a_garder_tracks} #On crée un sous dictionnaire en ne gardant que les clefs contenues dans la liste "à garder"
                
                d["featured_artists_id"] = [feat["id"] for feat in d["featured_artists"]]
                d["featured_artists_names"] = [feat["name"] for feat in d["featured_artists"]]
                d["writer_artists_id"] = [writer["id"] for writer in d["writer_artists"]]
                d["writer_artists_names"] = [writer["name"] for writer in d["writer_artists"]]
                d["primary_artists_id"] = [ar["id"] for ar in d["primary_artists"]]
                d["primary_artists_names"] = [ar["name"] for ar in d["primary_artists"]]
                del d["primary_artists"]
                del d["featured_artists"]
                del d["writer_artists"]
                
                d["album_id"] = a_id 
            
            results_tracks.append(d)

    return pd.DataFrame(results_tracks) #On renvoie la base de données correspondante
