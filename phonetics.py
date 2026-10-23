import frhyme
from typing import List, Dict, Tuple, Optional, T
import json
import os
from glob import glob
import re
import cmudict

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
    

def reduce_rhyme_fr(phons, glide=False):
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

def rhyme_phoneme_fr(line:str, reduce:bool=True, glide:bool=False, nbest:int=1) -> str:
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
        phons = list(dict.fromkeys(reduce_rhyme_fr(p, glide) for p in phons))
    if not phons:
        return None
    else:
        result = phons[0] if nbest == 1 else phons
    return result

# ---------------------- English treatment pipeline --------------------- #
#TBD

"""English rhyme phonemes for lines of text, using CMUdict (pip install cmudict)."""

# Words = letter runs (accents included) with inner apostrophes / hyphens.
_WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)
_STRESS = re.compile(r"\d$")

# word -> list of pronunciations, each one a list of ARPAbet phonemes,
# e.g. "moment" -> [["M", "OW1", "M", "AH0", "N", "T"]]
_CMU = cmudict.dict()

# ARPAbet semi-vowels: the "glide" leading a vowel, as in "few" = F Y UW1
_SEMI = {"Y", "W"}

def _is_vowel(phoneme: str) -> bool:
    return phoneme[-1].isdigit()  # only vowels carry a stress digit (0/1/2)

def _join(phons: list[str], keep_stress: bool) -> str:
    return " ".join(p if keep_stress else _STRESS.sub("", p) for p in phons)

def _pronunciations(word: str) -> list[list[str]]:
    """All CMUdict pronunciations of a word ([] if unknown)."""
    word = word.replace("’", "'")
    # exact word, then hyphens removed, then last part of a hyphenated word
    for candidate in (word, word.replace("-", ""), word.split("-")[-1]):
        if candidate in _CMU:
            return _CMU[candidate]
    return []

def reduce_rhyme_en(phons: list[str], glide: bool = False, keep_stress: bool = False) -> str:
    """
    Only keeps the rhyming part: from the last stressed vowel (primary or
    secondary stress) to the end of the word.
    ["M","OW1","M","AH0","N","T"]   -> "OW M AH N T"
    ["G","R","IY1","T","IH0","NG"]  -> "IY T IH NG"   (feminine rhyme: meeting/greeting)
    ["F","OW1","T","AH0","G","R","AE2","F"] -> "AE F"  (photograph/giraffe)
    No stressed vowel (e.g. "the"): falls back on the last vowel.
    glide=True also keeps the semi-vowel leading the first kept vowel
    (["F","Y","UW1"] -> "Y UW" instead of "UW").
    keep_stress=True keeps the stress digits ("AE2 F" instead of "AE F").
    """
    start = None
    for i in range(len(phons) - 1, -1, -1):
        if phons[i][-1] in "12":
            start = i
            break
    if start is None:  # no stressed vowel: last vowel
        for i in range(len(phons) - 1, -1, -1):
            if _is_vowel(phons[i]):
                start = i
                break
    if start is None:  # no vowel at all: keep everything
        start = 0
    if glide and start > 0 and phons[start - 1] in _SEMI:
        start -= 1
    return _join(phons[start:], keep_stress)


def rhyme_phoneme_en(
    line: str,
    reduce: bool = True,
    glide: bool = False,
    nbest: int = 1,
    keep_stress: bool = False,
) -> str | list[str] | None:
    """
    - nbest=1 : the rhyme of the best (first) pronunciation of the last word, e.g. "OW M AH N T"
    - nbest>1 : a list of the rhymes of the nbest first pronunciations (duplicates removed)
    - reduce=False : the full pronunciation instead of the rhyming part
    - None if the line does not contain any word, or if the last word is not in CMUdict.
    Phonemes are ARPAbet, separated by spaces; stress digits are dropped unless keep_stress.
    """
    word = _find_last_word(line)
    if word is None:
        return None
    prons = _pronunciations(word)[:nbest]
    if not prons:
        return None
    if reduce:
        phons = [reduce_rhyme_en(p, glide, keep_stress) for p in prons]
    else:
        phons = [_join(p, keep_stress) for p in prons]
    phons = list(dict.fromkeys(phons))
    return phons[0] if nbest == 1 else phons


# ----------------------- Main treatment pipeline ----------------------- #

def link_rhymes_per_song(song:Dict[str, T], output_path:Optional[str]) -> Dict[str, T]:
    lyrics = song["lyrics"]
    lang = song["lang"]
    sections = list()

    for section in lyrics:
        lines = section["lines"]
        updated_lines = list()

        for line in lines:
            text = line["text"]
            if (lang == "fr"):
                rhyme = rhyme_phoneme_fr(line=text)
            else:
                rhyme = rhyme_phoneme_fr(line=text)
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
                
