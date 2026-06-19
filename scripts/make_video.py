# make_video.py
# Génère une vidéo annotée à partir de résultats de tracking déjà calculés.
# Réutilise les fichiers .txt (format MOT) sans relancer la détection/suivi.
# Relativement rapide (quelques secondes) puisqu'il ne fait que dessiner.
#
# Usage :
#   python make_video.py --sequence data/MOT17/train/MOT17-02-DPM \
#                        --results results/ablation/v8_combo/MOT17-02-DPM.txt
#
#   # Si --results omis, cherche results/mot17/<nom_sequence>.txt
#   python make_video.py --sequence data/MOT17/train/MOT17-02-DPM

import sys
import argparse
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import MOT17Sequence, draw_tracks
import numpy as np


def load_tracks_by_frame(results_path: str) -> dict:
    """
    Charge un fichier de résultats MOT et l'indexe par numéro d'image.

    Format MOT : frame,id,x,y,w,h,conf,-1,-1,-1
    (x,y) = coin supérieur gauche, (w,h) = largeur/hauteur

    Retourne
    --------
    dict {frame_id: np.ndarray (N, 5)} où chaque ligne est
    [x1, y1, x2, y2, track_id] — le format attendu par draw_tracks().
    """
    tracks_by_frame = {}
    with open(results_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            # Conversion [x,y,w,h] → [x1,y1,x2,y2,id]
            row = [x, y, x + w, y + h, track_id]
            tracks_by_frame.setdefault(frame_id, []).append(row)

    # Convertir les listes en tableaux numpy
    return {fid: np.array(rows, dtype=np.float32)
            for fid, rows in tracks_by_frame.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Génère une vidéo annotée depuis des résultats MOT existants"
    )
    parser.add_argument(
        "--sequence",
        required=True,
        help="Chemin vers la séquence MOT17 (pour lire les frames)"
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Fichier de résultats .txt. Si absent, cherche dans results/mot17/"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Chemin de la vidéo de sortie .avi. Si absent, à côté des résultats."
    )
    args = parser.parse_args()

    seq_path = Path(args.sequence)
    sequence = MOT17Sequence(str(seq_path))

    # Localiser le fichier de résultats
    if args.results:
        results_path = Path(args.results)
    else:
        results_path = Path("results/mot17") / f"{sequence.name}.txt"

    if not results_path.exists():
        print(f"[ERREUR] Fichier de résultats introuvable : {results_path}")
        print("  Lancez d'abord le tracking, ou précisez --results")
        sys.exit(1)

    # Chemin de sortie
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_path.with_suffix(".avi")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Video] Séquence  : {sequence.name} ({len(sequence)} frames)")
    print(f"[Video] Résultats : {results_path}")
    print(f"[Video] Sortie    : {output_path}")

    # Charger les pistes
    tracks_by_frame = load_tracks_by_frame(str(results_path))

    # Initialiser l'écriture vidéo
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, sequence.fps,
        (sequence.img_width, sequence.img_height)
    )

    # Dessiner image par image
    for frame_id, frame_bgr in sequence:
        tracks = tracks_by_frame.get(frame_id, np.empty((0, 5), dtype=np.float32))
        annotated = draw_tracks(frame_bgr, tracks)
        writer.write(annotated)
        if frame_id % 100 == 0:
            print(f"  Frame {frame_id}/{len(sequence)}")

    writer.release()
    print(f"\n[Video]  Vidéo générée : {output_path}")


if __name__ == "__main__":
    main()
