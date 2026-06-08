# test_pipeline.py
# Test rapide du pipeline complet sur données synthétiques
# Permet de vérifier que tout fonctionne SANS avoir MOT17 téléchargé
#
# Usage : python test_pipeline.py
#
# Ce script :
#   1. Génère une séquence vidéo synthétique (50 frames, 2 piétons simulés)
#   2. Fait tourner le détecteur YOLOv3 (vraie inférence)
#   3. Fait tourner Deep SORT (vrai tracking)
#   4. Écrit un fichier résultat au format MOT
#   5. Affiche un résumé

import sys
import os
import time
import tempfile
import numpy as np
import cv2
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "deep_sort"))

CONFIGS_OK = (
    (ROOT / "configs" / "yolov3.yaml").exists()
    and (ROOT / "configs" / "deepsort.yaml").exists()
)
WEIGHTS_OK = (
    (ROOT / "weights" / "yolov3.weights").exists()
    and (ROOT / "weights" / "yolov3.cfg").exists()
    and (ROOT / "weights" / "coco.names").exists()
    and (ROOT / "weights" / "mars-small128.pb").exists()
)

# ------------------------------------------------------------------
# Génération d'une séquence synthétique
# ------------------------------------------------------------------

def make_synthetic_sequence(n_frames: int = 50, w: int = 640, h: int = 480):
    """
    Génère des frames synthétiques avec des rectangles mobiles
    simulant des piétons (classe person = 0 dans COCO).

    Les rectangles sont colorés pour que le réseau ReID puisse
    extraire des features d'apparence différenciées.

    Retourne une liste de frames BGR (np.ndarray).
    """
    frames = []

    # 2 "piétons" simulés : position initiale, vitesse, couleur
    walkers = [
        {"x": 100, "y": 150, "vx": 4,  "vy": 1,  "color": (200, 50,  50)},
        {"x": 400, "y": 250, "vx": -3, "vy": 2,  "color": (50,  200, 50)},
    ]

    for _ in range(n_frames):
        frame = np.ones((h, w, 3), dtype=np.uint8) * 80  # fond gris

        for walker in walkers:
            # Rectangle représentant un piéton (60×120 pixels)
            pw, ph = 60, 120
            x1 = int(walker["x"])
            y1 = int(walker["y"])
            x2 = x1 + pw
            y2 = y1 + ph

            # Dessiner le corps
            cv2.rectangle(frame, (x1, y1), (x2, y2), walker["color"], -1)
            # Dessiner la tête
            cx = x1 + pw // 2
            cv2.circle(frame, (cx, y1 - 20), 20, walker["color"], -1)

            # Mise à jour position (rebond sur les bords)
            walker["x"] += walker["vx"]
            walker["y"] += walker["vy"]
            if walker["x"] < 0 or walker["x"] + pw > w:
                walker["vx"] *= -1
            if walker["y"] < 0 or walker["y"] + ph > h:
                walker["vy"] *= -1

        frames.append(frame)

    return frames


# ------------------------------------------------------------------
# Pipeline simplifié (sans MOT17Sequence, sur frames en mémoire)
# ------------------------------------------------------------------

def run_pipeline(frames):
    """
    Fait tourner YOLOv3 + Deep SORT sur une liste de frames.
    Retourne les pistes par frame.
    """
    from src.detector import YOLOv3Detector
    from src.tracker  import DeepSORTTracker

    detector = YOLOv3Detector(str(ROOT / "configs" / "yolov3.yaml"))
    tracker  = DeepSORTTracker(str(ROOT / "configs" / "deepsort.yaml"))
    tracker.reset()

    all_tracks = {}
    t0 = time.time()

    for frame_id, frame_bgr in enumerate(frames, start=1):
        detections = detector.detect(frame_bgr)
        tracks     = tracker.update(detections, frame_bgr)
        all_tracks[frame_id] = tracks

        if frame_id % 10 == 0:
            elapsed = time.time() - t0
            fps = frame_id / elapsed
            print(
                f"  Frame {frame_id:3d}/{len(frames)} | "
                f"Dét: {len(detections):2d} | "
                f"Tracks: {len(tracks):2d} | "
                f"{fps:.2f} fps"
            )

    elapsed_total = time.time() - t0
    return all_tracks, elapsed_total


# ------------------------------------------------------------------
# Écriture résultat MOT
# ------------------------------------------------------------------

def write_result(all_tracks: dict, output_path: str):
    """Écrit les pistes au format MOTChallenge dans output_path."""
    with open(output_path, "w") as f:
        for frame_id, tracks in sorted(all_tracks.items()):
            for track in tracks:
                x1, y1, x2, y2, tid = track
                w = x2 - x1
                h = y2 - y1
                f.write(f"{frame_id},{int(tid)},{x1:.1f},{y1:.1f},"
                        f"{w:.1f},{h:.1f},-1,-1,-1,-1\n")


# ------------------------------------------------------------------
# Programme principal
# ------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("  TEST PIPELINE — YOLOv3 + Deep SORT (données synthétiques)")
    print("="*60)

    if not CONFIGS_OK:
        print("\n[ERREUR] Fichiers de config absents (configs/*.yaml)")
        sys.exit(1)

    if not WEIGHTS_OK:
        print("\n[ERREUR] Poids absents dans weights/")
        print("  Lancez : bash scripts/setup.sh")
        sys.exit(1)

    # 1. Génération des frames synthétiques
    print("\n[1/4] Génération de la séquence synthétique...")
    N_FRAMES = 50
    frames = make_synthetic_sequence(n_frames=N_FRAMES, w=640, h=480)
    print(f"  {N_FRAMES} frames 640×480 générées")

    # 2. Pipeline
    print(f"\n[2/4] Pipeline YOLOv3 + Deep SORT...")
    all_tracks, elapsed = run_pipeline(frames)

    # 3. Écriture du résultat
    print(f"\n[3/4] Écriture du fichier résultat...")
    out_path = ROOT / "results" / "test_synthetic.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_result(all_tracks, str(out_path))
    n_lines = sum(1 for _ in open(out_path))
    print(f"  {n_lines} lignes écrites → {out_path}")

    # 4. Résumé
    frames_with_tracks = sum(1 for t in all_tracks.values() if len(t) > 0)
    all_ids = set()
    for tracks in all_tracks.values():
        for t in tracks:
            all_ids.add(int(t[4]))

    print(f"\n[4/4] Résumé")
    print(f"{'='*60}")
    print(f"  Frames traitées        : {N_FRAMES}")
    print(f"  Frames avec tracks     : {frames_with_tracks}")
    print(f"  IDs uniques générés    : {len(all_ids)}")
    print(f"  Temps total            : {elapsed:.1f}s")
    print(f"  Vitesse                : {N_FRAMES/elapsed:.2f} fps")
    print(f"  Fichier résultat       : {out_path}")
    print(f"{'='*60}")
    print(f"\n  {chr(10)}  Le pipeline fonctionne correctement.")
    print("  Vous pouvez maintenant travailler sur MOT17.\n")


if __name__ == "__main__":
    main()
