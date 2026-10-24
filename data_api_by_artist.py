# Choper un token ici : https://genius.com/api-clients

import pandas as pd
from lyricsgenius import Genius
import numpy as np
from typing import Optional, Dict, T
import os
import json
from requests.exceptions import RequestException
import time

metadatas_a_garder_albums = ['_type', 'full_title', 'title', 'id', 'path', 'release_date_components', 'primary_artists']

def get_artist_id(artist_name:str) :
    artist = genius.search_artist(artist_name, get_full_info=False, max_songs=0)
    return artist.to_dict()["id"]

def get_all_albums(artist_name:str, access_token:str, output_folder:Optional[str]=None) -> pd.DataFrame:
    """Renvoie toutes les infos nécessaires sur les albums à partir du nom de l'artiste."""
    genius = Genius(access_token)

    artist = genius.search_artist(artist_name, get_full_info=False, max_songs=0) #On récupère la fiche de l'artiste pour avoir son id

    if artist == None :
        print("Nom d'artiste non reconnu")
        return None
    
    artist_id = artist.to_dict()["id"]
    try:
        albums = genius.artist_albums(artist_id)["albums"]
    except Exception as e:
        print("Erreur dans l'appel de récupération de l'album.\n" \
        f"{e}")
        return None

    results_albums = []
    not_treated = list()

    for a in albums : 
        try: 
            album_track = genius.album_tracks(a["id"])
        except Exception as e:
            print("Erreur dans l'appel de récupération des tracks de l'album.\n" \
                    f"{e}")
            not_treated.append(a["id"])
            continue

        a["tracks_id"] = [t["song"]["id"] for t in album_track["tracks"]]
        a = {p:a[p] for p in a if p in metadatas_a_garder_albums}
        a["primary_artists_id"] = [ar["id"] for ar in a["primary_artists"]]
        a["primary_artists_names"] = [ar["name"] for ar in a["primary_artists"]]
        del a["primary_artists"]
        results_albums.append(a)

        if (output_folder):
            albumdf = pd.DataFrame(a)
            albumdf.to_csv(os.path.join(output_folder, f"{a['id']}.csv"))

    return pd.DataFrame(results_albums) #On renvoie la base de données correspondante


metadatas_a_garder_tracks = ['primary_artists', 'full_title', 'title', 'id', 'path', 'featured_artists', 'writer_artists', 'lyrics', 'language']
# Leur colonne language est fucked up, je sais pas sur quoi elle s'appuie mais elle a détecté un morceau de Stupeflip comme du tchèque
def get_all_songs_albums(artist_name:str, access_token:str, k:int=1, input_df:Optional[pd.DataFrame]=None) -> pd.DataFrame: 
    """Renvoie tous les sons sortis par l'artiste, en y allant album par album."""
    genius = Genius(access_token)

    artist = genius.search_artist(artist_name, get_full_info=False, max_songs=0) #On récupère la fiche de l'artiste

    if artist == None :
        print("Nom d'artiste non reconnu")
        return None
    
    artist_id = artist.to_dict()["id"]
    try:
        albums = genius.artist_albums(artist_id)["albums"]
    except Exception as e:
        print("Erreur dans l'appel de récupération de l'album.\n" \
        f"{e}")
        return None
    
    albums_id = [a["id"] for a in albums]
    results_tracks = []

    if input_df is not None:
            albums_to_ignore = input_df["album_id"].unique()
            albums_id = [id for id in albums_id if id not in albums_to_ignore]

    try:
        for i_album, a_id in enumerate(albums_id[:k]) : 
            try:
                tracks_id = [t["song"]["id"] for t in genius.album_tracks(a_id)["tracks"]]
            except Exception as e:
                print(f"Erreur dans le load des tracks de {a_id}.\n"
                    f"{e}")
                continue

            print(tracks_id)
            album_tracks= list()
            for i_track, t_id in enumerate(tracks_id) : 
                print(t_id)
                song = None
                for attempt in range(3):
                    try :  #Si le titre n'est pas transcrit, on a une erreur
                        song = genius.search_song(song_id=t_id)
                        break
                    except AssertionError as ae :
                        print(ae, f"Erreur concernant le titre n°{i_track} (en comptant 0), ayant pour id {t_id}") 
                        break
                    except RequestException as e:
                        print(f"Timeout titre {t_id} (essai {attempt+1}/3) : {e}")
                        time.sleep(2 ** attempt)
                else: # si la boucle a tourné jusqu'au bout sans break, on entre dedans
                    raise ConnectionError(f"Abandon sur le titre {t_id}")

                if song : #Il arrive que le song soit None, c'est entièrement la faute du référencement de l'API
                    d = song.to_dict() #On convertit le titre en dictionnaire
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
                    album_tracks.append(d)
                
                results_tracks.extend(album_tracks)
    except (KeyboardInterrupt, ConnectionError, RequestException) as e:
        print(f"Interruption ({type(e).__name__}) : on sauvegarde {len(results_tracks)} titres déjà récupérés.")
        
        output_df = pd.DataFrame(results_tracks)
        result = output_df if input_df is None else pd.concat([input_df, output_df], ignore_index=True)
        result.to_csv("backup.csv", index=False)  
        return result


    output_df = pd.DataFrame(results_tracks)
    return  pd.concat([input_df, output_df]) if input_df else output_df#On renvoie la base de données correspondante

def retrieve_release_metadata(release:Dict[str, T]) -> Dict[str, T]:
    release_recap = {
        "release_id": release["id"],
        "title": release["name"],
        "release_date": release["release_date_components"],
        "artist_ids": release["artist"]["id"],
    }
    return release_recap

def get_releases_metadata(artist_name:str, access_token:str, output_folder:Optional[str]=None) -> pd.DataFrame:
    genius = Genius(access_token)
    artist = genius.search_artist(artist_name, get_full_info=False, max_songs=0) #On récupère la fiche de l'artiste pour avoir son id

    if artist == None :
            print("Nom d'artiste non reconnu")
            return None
        
    artist_id = artist.to_dict()["id"]

    try:
        albums = genius.artist_albums(artist_id).get("albums", None)

    except Exception as e:
        print("Erreur dans l'appel de récupération de l'album.\n" \
        f"{e}")
        return None

    if (output_folder):
        os.makedirs(output_folder, exist_ok=True)

    album_recaps = list()
    for album in albums:
        album_recap = retrieve_release_metadata(album)
        try:
            id = album_recap["release_id"]
            output_path = os.path.join(output_folder, f"{id}.json")
            with open(output_path, "w") as file:
                json.dump(album_recap, file)
        except Exception as e:
            print(f"Erreur dans l'enregistrement des metadonnées de la release {id}.\n"
                  f"{e}")
            
        album_recaps.append(album_recap)
    
    return pd.DataFrame(album_recaps)

