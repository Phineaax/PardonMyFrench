#!/usr/bin/env python3
"""Supprime les lignes dont persona_id vaut "none" dans des fichiers JSON de paroles.

Exemples :
    python remove_none_lines.py ./paroles --dry-run
    python remove_none_lines.py ./paroles
    python remove_none_lines.py ./paroles -r --out ./paroles_clean
    python remove_none_lines.py ./paroles --backup --include-null
"""
import argparse
import json
import shutil
import sys
from pathlib import Path


def is_none_persona(value, include_null):
    if isinstance(value, str):
        return value.strip().lower() == "none"
    if include_null:
        return value is None
    return False


def clean_song(data, include_null=False, renumber=True, keep_empty_sections=False):
    """Nettoie un dict de chanson. Retourne (lignes_supprimees, sections_supprimees)."""
    removed_lines = 0
    removed_sections = 0
    lyrics = data.get("lyrics")
    if not isinstance(lyrics, list):
        return 0, 0

    new_sections = []
    for section in lyrics:
        lines = section.get("lines") if isinstance(section, dict) else None
        if not isinstance(lines, list):
            new_sections.append(section)
            continue

        kept = [
            ln for ln in lines
            if not (isinstance(ln, dict)
                    and is_none_persona(ln.get("persona_id"), include_null))
        ]
        removed_lines += len(lines) - len(kept)

        if renumber and len(kept) != len(lines):
            for i, ln in enumerate(kept):
                if isinstance(ln, dict) and "n" in ln:
                    ln["n"] = str(i)

        section["lines"] = kept
        if not kept and not keep_empty_sections and len(lines) > 0:
            removed_sections += 1
            continue
        new_sections.append(section)

    data["lyrics"] = new_sections
    return removed_lines, removed_sections


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path, help="Dossier contenant les fichiers .json")
    p.add_argument("-r", "--recursive", action="store_true",
                   help="Parcourir aussi les sous-dossiers")
    p.add_argument("--out", type=Path, default=None,
                   help="Dossier de sortie (sinon les fichiers sont modifiés sur place)")
    p.add_argument("--backup", action="store_true",
                   help="Crée un .bak avant de modifier sur place")
    p.add_argument("--dry-run", action="store_true",
                   help="Affiche ce qui serait supprimé sans rien écrire")
    p.add_argument("--include-null", action="store_true",
                   help="Supprime aussi les lignes dont persona_id est null/absent")
    p.add_argument("--keep-n", action="store_true",
                   help="Ne pas renuméroter le champ 'n' après suppression")
    p.add_argument("--keep-empty-sections", action="store_true",
                   help="Garder les sections devenues vides")
    args = p.parse_args()

    if not args.folder.is_dir():
        sys.exit(f"Dossier introuvable : {args.folder}")

    pattern = "**/*.json" if args.recursive else "*.json"
    files = sorted(args.folder.glob(pattern))
    if not files:
        sys.exit("Aucun fichier .json trouvé.")

    total_files = total_lines = total_sections = 0
    for path in files:
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[IGNORÉ] {path} : {e}")
            continue
        if not isinstance(data, dict):
            continue

        lines, sections = clean_song(
            data,
            include_null=args.include_null,
            renumber=not args.keep_n,
            keep_empty_sections=args.keep_empty_sections,
        )
        if lines == 0:
            continue

        total_files += 1
        total_lines += lines
        total_sections += sections
        print(f"[OK] {path} : {lines} ligne(s) supprimée(s)"
              + (f", {sections} section(s) vide(s) retirée(s)" if sections else ""))

        if args.dry_run:
            continue

        if args.out:
            dest = args.out / path.relative_to(args.folder)
            dest.parent.mkdir(parents=True, exist_ok=True)
        else:
            dest = path
            if args.backup:
                shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))

        with dest.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.write("\n")

    mode = " (simulation)" if args.dry_run else ""
    print(f"\nTerminé{mode} : {total_lines} ligne(s) supprimée(s), "
          f"{total_sections} section(s) retirée(s), {total_files} fichier(s) modifié(s) "
          f"sur {len(files)}.")


if __name__ == "__main__":
    main()
