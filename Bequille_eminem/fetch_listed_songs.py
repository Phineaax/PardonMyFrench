#!/usr/bin/env python3
"""
Récupère sur Genius uniquement les titres de la liste Eminem / Slim Shady
(définie dans tag_personas.py) et les sauvegarde avec un champ "persona_id".

Principe : on parcourt les albums de l'artiste et la liste de leurs pistes
(léger : titres + ids, sans paroles), puis on ne télécharge les paroles
(genius.search_song, lent) que pour les titres de la liste.

Usage :
    export GENIUS_TOKEN="..."
    python fetch_listed_songs.py                  # artiste "Eminem"
    python fetch_listed_songs.py -o Eminem_songs
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from requests.exceptions import RequestException

from tag_personas import PERSONAS, TITLE_TO_PERSONA, normalize
import re

HEADER_RE = re.compile(r"^\[(.*?)\]$")
NOISE_RE = re.compile(r"^(\d*Embed|You might also like)$")


def _section_type(label: str) -> str:
    """'Verse 1' -> 'verse', 'Pre-Chorus' -> 'pre-chorus'."""
    label = re.sub(r"\s*\d+\s*$", "", label.strip().lower())
    return re.sub(r"\s+", "-", label) or "unknown"


def _performer(raw: str) -> str:
    """'Eminem & Dr. Dre' -> 'Eminem, Dr. Dre'."""
    names = [n.strip() for n in re.split(r"[,&]|\band\b", raw) if n.strip()]
    return ", ".join(names)


def parse_lyrics(text: str, persona: str) -> list:
    """Texte brut Genius ('[Verse 1: Eminem]' + lignes) -> sections au format cible."""
    sections, current = [], None
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or NOISE_RE.match(line):
            continue
        header = HEADER_RE.match(line)
        if header:
            label, _, perf = header.group(1).partition(":")
            current = {"type": _section_type(label), "performer": _performer(perf), "lines": []}
            sections.append(current)
            continue
        if current is None:
            if line.endswith("Lyrics") or "Contributor" in line:
                continue  # en-tête parasite "N ContributorsTitle Lyrics"
            current = {"type": "unknown", "performer": "", "lines": []}
            sections.append(current)
        current["lines"].append(
            {"n": str(len(current["lines"])), "text": line, "persona_id": persona}
        )

    sections = [s for s in sections if s["lines"]]
    return [
        {"section_id": f"section_{i}", "type": s["type"], "performer": s["performer"], "lines": s["lines"]}
        for i, s in enumerate(sections)
    ]


def to_song_json(d: dict, persona: str, album_id: int) -> dict:
    """song.to_dict() de lyricsgenius -> JSON final (même schéma que vos fichiers)."""
    return {
        "song_id": d["id"],
        "title": d["title"],
        "language": d.get("language"),
        "lyricists": [w["id"] for w in d.get("writer_artists") or []],
        "lyrics": parse_lyrics(d.get("lyrics"), persona),
        "album_id": album_id,
    }


def _paginate(fetch, key, **kwargs):
    """Itère sur les pages d'un endpoint lyricsgenius jusqu'à une page vide."""
    page = 1
    while True:
        res = fetch(per_page=50, page=page, **kwargs)
        items = res.get(key, [])
        if not items:
            return
        yield from items
        page = res.get("next_page") or None
        if page is None:
            return


def find_listed_tracks(genius, artist_id: int) -> dict:
    """{id_chanson: (persona, titre)} pour les titres de la liste, sans paroles."""
    found = {}          # id -> (persona, titre)
    seen_titles = set() # un seul exemplaire par titre (évite live / deluxe / remaster)
    for album in _paginate(genius.artist_albums, "albums", artist_id=artist_id):
        try:
            tracks = list(_paginate(genius.album_tracks, "tracks", album_id=album["id"]))
        except Exception as e:
            print(f"[album {album['id']}] erreur : {e}")
            continue
        for t in tracks:
            song = t["song"]
            key = normalize(song["title"])
            if key in TITLE_TO_PERSONA and key not in seen_titles:
                seen_titles.add(key)
                found[song["id"]] = (TITLE_TO_PERSONA[key], song["title"], album["id"])
    return found


def fetch_song(genius, song_id: int):
    """Télécharge une chanson avec les mêmes retries que votre code."""
    for attempt in range(3):
        try:
            return genius.search_song(song_id=song_id)
        except AssertionError as e:
            print(f"[{song_id}] titre non transcrit : {e}")
            return None
        except RequestException as e:
            print(f"[{song_id}] timeout (essai {attempt + 1}/3) : {e}")
            time.sleep(2 ** attempt)
    raise ConnectionError(f"Abandon sur le titre {song_id}")


def collect_listed_songs(genius, artist_name: str = "Eminem", output_dir=None) -> list:
    artist = genius.search_artist(artist_name, get_full_info=False, max_songs=0)
    if artist is None:
        print("Nom d'artiste non reconnu")
        return []
    artist_id = artist.to_dict()["id"]

    candidates = find_listed_tracks(genius, artist_id)
    print(f"{len(candidates)} titre(s) de la liste trouvé(s) sur les albums")

    rows = []
    for song_id, (persona, title, album_id) in candidates.items():
        song = fetch_song(genius, song_id)
        if not song:
            continue
        row = to_song_json(song.to_dict(), persona, album_id)
        rows.append(row)
        print(f"[{persona}] {title}")
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / f"{song_id}.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    found_keys = {normalize(title) for _, title, _ in candidates.values()}
    missing = [
        t for titles in PERSONAS.values() for t in titles
        if normalize(t) not in found_keys
    ]
    if missing:
        print("\nNon trouvés sur les albums de l'artiste :", ", ".join(missing))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artist", default="Eminem")
    parser.add_argument("-o", "--output-dir", default="Eminem_songs")
    args = parser.parse_args()

    token = os.environ.get("GENIUS_TOKEN")
    if not token:
        raise SystemExit("Définissez la variable d'environnement GENIUS_TOKEN.")

    from lyricsgenius import Genius  # import ici pour pouvoir tester sans la lib

    genius = Genius(token, timeout=15, sleep_time=0.5)
    rows = collect_listed_songs(genius, args.artist, args.output_dir)
    print(f"\n{len(rows)} chanson(s) sauvegardée(s) dans {args.output_dir}/")


if __name__ == "__main__":
    main()