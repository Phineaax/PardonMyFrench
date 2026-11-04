import re
import json
from typing import List, Dict, Tuple, Optional, T
import pandas as pd

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

def count_tokens(line:str, lang:str) -> int:
    if (lang == "fr"):
        regex = TOKENS_FR_REG
    elif (lang == "en"):
        regex = TOKENS_ENG_REG
    else:
        print(f"Selected language is not understood: {lang}. Please choose between fr and en.\n"
              "French chosen by default.")

    tokens = re.findall(pattern=regex, string=line)
    return len(tokens)


def make_lyrics_dataframe(artist_dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for i in range(len(artist_dataframe)):
        song_id = artist_dataframe["song_id"].iloc[i]
        lyrics_obj = artist_dataframe["lyrics"].iloc[i]

        sections = lyrics_obj if isinstance(lyrics_obj, list) else [lyrics_obj]

        for section in sections:
            for line in section.get("lines", []):
                rows.append({
                    "song_id": song_id,
                    "section_id": section.get("section_id"),
                    "performer_id": line.get("performer_id"),
                    "text": line.get("text"),
                })

    return pd.DataFrame(rows)

def chunk(lyrics_dataframe:str, chunk_tokens_nb:int=1500, min_chunk_size:int=MIN_CHUNK_SIZE) -> List[List[str]]:

    lines = lyrics_dataframe["text"].to_list()
    chunks = chunk(lines, chunk_tokens_nb=chunk_tokens_nb, min_chunk_size=min_chunk_size)

    chunks = list()
    token_count = 0
    current_chunk = list()
    for line in lines:
        if (token_count >= chunk_tokens_nb):
            chunks.append(current_chunk)
            current_chunk = list()
            token_count = 0

        token_count += count_tokens(line)
        current_chunk.append(line)

    if (token_count > min_chunk_size):
        chunks.append(current_chunk)

    return chunks