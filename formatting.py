import json
import re
from typing import List, Dict, Tuple, T, Optional
import pandas as pd
import os
import numpy as np

SECTION_TAGS = [r"Couplet(?:\s(?:\d+|unique))?", r"Interlude", r"Intro", r"Pont",
                r"Refrain", r"Post-refrain", r"Pre-refrain", r"Outro",
                r"Verse", r"Bridge", r"Chorus", r"Pre-Chorus"]

def with_num(p):   # numéro optionnel, sauf pour Couplet qui a déjà le sien
    return p if p.startswith("Couplet") else rf"{p}(?:\s\d+)?"

ANY_TAG = re.compile(r"^\s*\[(?P<tag>[^\]:]+?)\s*(?::\s*[^\]]*)?\]\s*$", re.M)
TAG = "|".join(with_num(p) for p in SECTION_TAGS)
KNOWN = re.compile(TAG, re.I)
IGNORED = re.compile(r'Paroles de\b.*', re.I)

PAIR = re.compile(
        rf"""
        \[(?P<tag>{TAG})(?:\s*:\s*(?P<who>[^\]]+))?\]
        (?P<body>.*?)
        (?=\[(?:{TAG})(?:\s*:\s*[^\]]+)?\]|\Z)
        """,
        re.S | re.X | re.I,
    )

ARTIST_DICT = {
    "19547": "Népal",
    "60891": "Di-Meh",
    "13063": "Nekfeu",
    "41297": "Sheldon",
    "13259": "Doums",
    "14648": "3010"
}

## ------------------- HELPERS ------------------- ##
def unknown_tags(lyrics: List[str]) -> bool:
    found = {m["tag"] for t in lyrics for m in ANY_TAG.finditer(t)}
    raw_unknown = {t for t in found
                   if not KNOWN.fullmatch(t) and not IGNORED.fullmatch(t)}
    if raw_unknown:
        print(f"{len(raw_unknown)} tags inconnus.\n"
              "Vérifier leur pertinence et format, "
              "les ajouter à SECTION_TAGS si pertinents.\n"
              f"Tags inconnus: {[repr(t) for t in raw_unknown]}")
    return bool(raw_unknown)
    
    
def make_lines_dict(lines: List[str]) -> List[Dict[str,str]]:
    cleaned_lines = [line for line in lines if len(line.strip()) > 0]
    formatted_lines = list()
    for (idx, line) in enumerate(cleaned_lines):
        line_dict = {
            "n": f"{idx}",
            "text": line
        }
        formatted_lines.append(line_dict)
    return formatted_lines


def lyricsparser(lyrics: str) -> List[Dict[str, T]]:
    new_tags = unknown_tags([lyrics])
    if (new_tags):
        return list()

    lyrics_list = list()
    for (idx, section) in enumerate(PAIR.finditer(lyrics)):
        tag = section["tag"]
        if ("Paroles de" in tag):
            continue

        sec_type = tag.split(sep=" ")[0].lower()
        body = section["body"]
        lines = body.split("\n")

        formatted_lines = make_lines_dict(lines=lines)

        try:
            performers = section["who"].split(sep=" ")
            reversed_artist_dict = {perf: id for (perf, id) in zip(ARTIST_DICT.values(), ARTIST_DICT.keys())}

            perf_ids = ", ".join([reversed_artist_dict.get(performer, "") for performer in performers])
        except:
            perf_ids = []
            
        sec_recap = {
                     "section_id": f"section_{idx}", 
                     "type": sec_type,
                     "performer": perf_ids,
                     "lines": formatted_lines
                     }
        lyrics_list.append(sec_recap)
    
    return lyrics_list

def single_song_formatter(row: Dict[str,T], output_path:Optional[str]=None) -> Dict[str,T]:
    lyrics = lyricsparser(row["lyrics"])
    song_recap = {
        "song_id": row["id"],
        "title": row["title"],
        "language": row["language"],
        "lyricists": row["writer_artists_id"],
        "lyrics": lyrics
    }

    if (output_path):
        try:
            with open(output_path, "w") as file:
                json.dump(song_recap, file)
        except Exception as e:
            print(f"Erreur pour l'enregistrement de la chanson {row['id']}: {e}")   

    return song_recap

def bunch_of_songs_formatter(df: pd.DataFrame, output_folder:Optional[str]=None, k = np.inf) -> List[Dict[str,T]]:
    if (output_folder):
        os.makedirs(output_folder, exist_ok=True)

    i = 0
    all_songs = list()
    for _, row in df.iterrows():
        if (i >= k):
            break
        song_id = row["id"]
        song_output_path = os.path.join(output_folder, f"{song_id}.json") or None
        song_recap = single_song_formatter(row=row, output_path=song_output_path)
        all_songs.append(song_recap)
        i += 1

    return all_songs