import frhyme
from typing import List, Dict, Tuple, Optional, T
import json
import os
from glob import glob
import re

import frhyme

### Document for rhyme calculation #### 

# Words = sequence of letters (accents included) with apostrophes/hyphens.

# ---------------------- French treatment pipeline ---------------------- #
_WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)

_VOWELS = set("aeiouyEO29@#$()")
_SEMI = set("jw8")

def _find_last_word(line:str) -> str:
    mots = _WORD.findall(line)
    return mots[-1].lower() if mots else None
    

def reduce_rhyme(phons, glide=False):
    """
    Only keeps the rhyme from last syllable (last vowel and what follows)
 
    "mom#" -> "#", "om#s" -> "#s", "vjEn" -> "En", "uZuR" -> "uR".
    glide=True also keeps the semi-vowel leading last vowel.
    ("Rwa" -> "wa" instead of "a", "mj2" -> "j2" instead of "2").
    """
    p = phons
    # final mute e does not count for rhyme 
    if p.endswith("@") and any(c in _VOWELS for c in p[:-1]):
        p = p[:-1]
    for i in range(len(p) - 1, -1, -1):
        if p[i] in _VOWELS:
            if glide and i > 0 and p[i - 1] in _SEMI:
                i -= 1
            return p[i:]
    return p  # no vowel found, resent as it is

def rhyme_phoneme(line:str, reduce:bool=True, glide:bool=False, nbest:int=1) -> str:
    """
    - nbest=1 : one string per line best pronunciation, ex. "zO$"
    - nbest>1 : a list of best pronunciations per line
    - None if the line does not contain any word.
    Phonemes are represented in X-SAMPA; one character per phoneme.
    (A~ -> #, O~ -> $, E~ -> ), 9~ -> ( ).
    """
    result = []
    
    word = _find_last_word(line)
    if word is None:
        return None
    
    candidats = frhyme.lookup(word, nbest)  # list of (score, phonemes)
    phons = [phon for _, phon in candidats]
    if (reduce):
        phons = list(dict.fromkeys(reduce_rhyme(p, glide) for p in phons))
    if not phons:
        return None
    else:
        result = phons[0] if nbest == 1 else phons
    return result

# ---------------------- English treatment pipeline --------------------- #
#TBD


# ----------------------- Main treatment pipeline ----------------------- #

def link_rhymes_per_song(song:Dict[str, T], output_path:Optional[str]) -> Dict[str, T]:
    lyrics = song["lyrics"]
    sections = list()

    for section in lyrics:
        lines = section["lines"]
        updated_lines = list()

        for line in lines:
            text = line["text"]
            rhyme = rhyme_phoneme(line=text)
            line["rhyme"] = rhyme
            updated_lines.append(line)

        section["lines"] = updated_lines
        sections.append(section)

    song["lyrics"] = sections
    if (output_path):
        try:
            with open(output_path, "w") as file:
                json.dump(song, file)
        except:
            print("Chemin de sortie invalide. L'enregistrement n'a pas été réalisé.")

    return song

def update_songs_with_rhymes(songs:List[Dict[str, T]], output_folder:Optional[str]=None) -> List[Dict[str, T]]:
    if (output_folder):
        os.makedirs(output_folder, exist_ok=True)

    updated_songs = list()
    for song in songs:
        song_id = song["song_id"]
        if (output_folder):
            output_path = os.path.join(output_folder, f"{song_id}.json")
        else:
            output_path = None
        updated_songs.append(link_rhymes_per_song(song=song, output_path=output_path))

    return updated_songs

def read_songs_from_path(input_folder:str) -> List[Dict[str, T]]:
    songs = list()
    try:
        for input_path in glob(os.path.join(input_folder, "*.json")):
            with open(input_path, "r") as file:
                songs.append(json.load(file))
    except Exception as e:
        print("Erreur dans la lecture d'un fichier.\n" \
        f"{e}")

    return songs
                
