#!/usr/bin/env python3
"""
Ajoute un champ "persona_id" ("eminem" ou "slim_shady") à chaque ligne des
paroles des fichiers JSON d'un dossier, selon le titre de la chanson.
Les fichiers dont le titre n'est dans aucune des deux listes sont ignorés.

Usage :
    python tag_personas.py DOSSIER_ENTREE [-o Eminem_songs]
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

PERSONAS = {
    "eminem": [
        "Lose Yourself",
        "Till I Collapse",
        "Mockingbird",
        "When I'm Gone",
        "Stan",
        "The Monster",
        "Love the Way You Lie",
        "Sing for the Moment",
        "Cleanin' Out My Closet",
        "Beautiful",
        "Not Afraid",
        "Headlights",
        "Lucky You",
    ],
    "slim_shady": [
        "The Real Slim Shady",
        "Kim",
        "Guilty Conscience",
        "My Name Is",
        "Crack a Bottle",
        "Kill You",
        "Role Model",
        "'97 Bonnie & Clyde",
        "Just Don't Give a Fuck",
        "Just Don't Give a F***",
        "Without Me",
        "Rap God",
        "Houdini",
    ],
}


def normalize(title: str) -> str:
    """Minuscules, sans accents, sans (feat. ...) / [..], sans ponctuation."""
    t = unicodedata.normalize("NFKD", title)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower().replace("&", " and ")
    t = re.sub(r"[\(\[].*?[\)\]]", " ", t)  # (feat. X), [Remix]...
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return t.strip()


TITLE_TO_PERSONA = {
    normalize(title): persona
    for persona, titles in PERSONAS.items()
    for title in titles
}


def tag_song(data: dict, persona: str) -> dict:
    for section in data.get("lyrics", []):
        for line in section.get("lines", []):
            line["persona_id"] = persona
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Dossier contenant les .json")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("Eminem_songs"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    kept = skipped = errors = 0

    for path in sorted(args.input_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[erreur] {path.name}: {e}")
            errors += 1
            continue

        persona = TITLE_TO_PERSONA.get(normalize(str(data.get("title", ""))))
        if persona is None:
            skipped += 1
            continue

        out = args.output_dir / path.name
        out.write_text(
            json.dumps(tag_song(data, persona), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[{persona}] {data['title']} -> {out}")
        kept += 1

    print(f"\n{kept} fichier(s) écrit(s), {skipped} ignoré(s), {errors} erreur(s).")


if __name__ == "__main__":
    main()
