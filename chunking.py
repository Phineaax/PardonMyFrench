import re
import json
from typing import List, Dict, Tuple, Optional, T
import pandas as pd
import os

MIN_CHUNK_SIZE = 900

TOKENS_FR_REG = re.compile(
    r"""
    aujourd['’]hui                                # cas particulier
  | (?:qu|lorsqu|puisqu|jusqu|[cdjlmnst])['’](?=\w)   # élisions : l', j', qu', ...
  | \d+(?:[.,]\d+)*                               # nombres : 3,14  1.000
  | \w+(?:-\w+)*                                  # mots, y compris composés : peut-être, c'est-à-dire
  | [^\w\s]                                       # ponctuation
    """,
    re.IGNORECASE | re.VERBOSE,
)

TOKENS_ENG_REG = re.compile(
    r"""
    \d+(?:[.,]\d+)*                                # nombres : 3.14  1,000
  | [A-Za-z]+(?:['’][A-Za-z]+)*(?:-[A-Za-z]+)*     # mots, contractions (don't), composés (well-known)
  | [^\w\s]                                        # ponctuation
    """,
    re.VERBOSE,
)

def count_tokens(line: str, lang: str = "fr") -> int:
    if lang == "fr":
        regex = TOKENS_FR_REG
    elif lang == "en":
        regex = TOKENS_ENG_REG
    else:
        print(f"Selected language is not understood: {lang}. "
              "Please choose between fr and en.\nFrench chosen by default.")
        regex = TOKENS_FR_REG
    return len(regex.findall(line))


def chunk(
    lyrics_dataframe: pd.DataFrame,
    chunk_tokens_nb: int = 1500,
    min_chunk_size: int = MIN_CHUNK_SIZE,
    lang: str = "fr",
) -> List[List[str]]:
    lines = lyrics_dataframe["text"].dropna().to_list()

    chunks = []
    current_chunk = []
    token_count = 0

    for line in lines:
        if token_count >= chunk_tokens_nb:
            chunks.append(current_chunk)
            current_chunk = []
            token_count = 0

        token_count += count_tokens(line, lang)
        current_chunk.append(line)

    # Dernier chunk : on le garde s'il est assez gros,
    # sinon on le fusionne avec le précédent
    if current_chunk:
        if token_count >= min_chunk_size or not chunks:
            chunks.append(current_chunk)
        else:
            chunks[-1].extend(current_chunk)

    return chunks

def chunks_to_big_list(chunks) :
    return sum(chunks, start=[])

def repare_cyrillic_error(t) : 
    H = str.maketrans("аеорсухіјАЕОРСХТНКМВ", "aeopcyxijAEOPCXTHKMB")
    return t.translate(H)

def make_lyrics_dataframe(artist_dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for song_id, sections in zip(artist_dataframe["song_id"], artist_dataframe["lyrics"]):
        for section in sections:
            for line in section.get("lines", []):
                rows.append({
                    "song_id": song_id,
                    "section_id": section.get("section_id"),
                    "persona_id": line.get("persona_id"),
                    "text": repare_cyrillic_error(line.get("text")),
                })

    return pd.DataFrame(rows)

def lyrics_to_df(lyrics_json) :
    x = lyrics_json
    d = pd.DataFrame([{'song_id' : x['song_id'], 'song_title' : x['title'], 'lyricists_id' : x['lyricists'],  'language' : x['language'], 
  'section_id' : section['section_id'], 'section_type' : section['type'], 
  'n_line' : line['n'], 'persona_id' : line['persona_id'], 'text' : line['text']} 
  for section in x['lyrics'] for line in section['lines']])
    return d

def artist_folder_to_artist_df(json_folder_path) :
    total_data = pd.DataFrame()
    for s in os.listdir(json_folder_path) :
        print(s)
        with open(f"{json_folder_path}/{s}", 'r') as f :
            x = json.load(f)
        
        try :
            d=lyrics_to_df(x)
        except : 
            print(f"problème avec le son {s}")
            continue;
        total_data = pd.concat([total_data, d])

    return total_data