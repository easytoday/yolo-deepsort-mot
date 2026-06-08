# run_tracker.py
# Script principal : traitement d'une séquence MOT17 unique
# Usage : python run_tracker.py --sequence data/MOT17/train/MOT17-02-DPM
#         python run_tracker.py --sequence data/MOT17/train/MOT17-02-DPM --detector yolov8
#         python run_tracker.py --sequence data/MOT17/train/MOT17-02-DPM --visualize

import sys
import os
import argparse
import time
import cv2
from pathlib import Path

# Ajout des dossiers externes dans sys.path (deep_sort cloné par setup.sh)
sys.path.insert(0, str(Path(__file__).parent / "deep_sort"))

from src.detector_factory import build_detector
from src.tracker          import DeepSORTTracker
from src.utils            import MOT17Sequence, MOTResultWriter, draw_tracks


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv3/YOLOv8 + Deep SORT — Tracking sur une séquence MOT17"
    )
    parser.add_argument(
        "--sequence",
        required=True,
        help="Chemin vers la séquence MOT17 (ex: data/MOT17/train/MOT17-02-DPM)"
    )
    parser.add_argument( # ajout detecteur yolov8 -- ziZa
        "--detector",
        choices=["yolov3", "yolov8"],
        default="yolov3",
        help="Détecteur à utiliser (défaut: yolov3)"
    )
    parser.add_argument(
        "--detector-config",
        default=None,
        help="Fichier de config détecteur. Si absent, utilise le type par défaut" 
             "choisi (configs/yolov3.yaml ou configs/yolov8.yaml)"
    )
    parser.add_argument(
        "--tracker-config",
        default="configs/deepsort.yaml",
        help="Fichier de config Deep SORT (défaut: configs/deepsort.yaml)"
    )
    parser.add_argument(
        "--output-dir",
        default="results/mot17",
        help="Dossier de sortie pour les fichiers résultats (défaut: results/mot17)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Afficher les frames annotées pendant le traitement (lent sur CPU)"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Sauvegarder une vidéo annotée dans output-dir"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Initialisation du détecteur et du tracker
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"  {args.detector.upper()} + Deep SORT — MOT17 Tracker")
    print("="*60)

    # La factory choisit YOLOv3 ou YOLOv8 selon --detector.
    # Si --detector-config n'est pas fourni, elle prend le type par défaut .
    detector = build_detector(args.detector, args.detector_config)
    tracker  = DeepSORTTracker(args.tracker_config)
    tracker.reset()  # S'assurer que les IDs repartent de 1

    # ------------------------------------------------------------------
    # 2. Chargement de la séquence
    # ------------------------------------------------------------------
    sequence = MOT17Sequence(args.sequence)

    # ------------------------------------------------------------------
    # 3. Préparation des sorties
    # ------------------------------------------------------------------
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / f"{sequence.name}.txt"

    video_writer = None
    if args.save_video:
        video_path = output_dir / f"{sequence.name}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writer = cv2.VideoWriter(
            str(video_path), fourcc, sequence.fps,
            (sequence.img_width, sequence.img_height)
        )
        print(f"[Video] Sauvegarde → {video_path}")

    # ------------------------------------------------------------------
    # 4. Boucle principale : frame par frame
    # ------------------------------------------------------------------
    print(f"\n[Run] Traitement de {sequence.name}...")
    print(f"      {len(sequence)} frames | détecteur={args.detector} | CPU\n")

    total_detections = 0
    total_tracks     = 0
    time_start       = time.time()

    with MOTResultWriter(str(result_file)) as writer:
        for frame_id, frame_bgr in sequence:

            # --- Détection (YOLOv3 ou YOLOv8 selon le choix) ---
            detections = detector.detect(frame_bgr)
            total_detections += len(detections)

            # --- Suivi Deep SORT ---
            tracks = tracker.update(detections, frame_bgr)
            total_tracks += len(tracks)

            # --- Écriture au format MOT ---
            writer.write(frame_id, tracks)

            # --- Affichage progression ---
            if frame_id % 50 == 0 or frame_id == 1:
                elapsed  = time.time() - time_start
                fps_proc = frame_id / elapsed if elapsed > 0 else 0
                eta_s    = (len(sequence) - frame_id) / fps_proc if fps_proc > 0 else 0
                print(
                    f"  Frame {frame_id:4d}/{len(sequence)} | "
                    f"Dét: {len(detections):2d} | "
                    f"Tracks: {len(tracks):2d} | "
                    f"{fps_proc:.1f} fps | "
                    f"ETA: {eta_s:.0f}s"
                )

            # --- Visualisation optionnelle ---
            if args.visualize or args.save_video:
                frame_annotated = draw_tracks(frame_bgr, tracks)

                if args.visualize:
                    cv2.imshow(f"{args.detector} + Deep SORT", frame_annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[Visualize] Interruption par l'utilisateur")
                        break

                if video_writer is not None:
                    video_writer.write(frame_annotated)

    # ------------------------------------------------------------------
    # 5. Résumé
    # ------------------------------------------------------------------
    elapsed_total = time.time() - time_start
    avg_fps       = len(sequence) / elapsed_total if elapsed_total > 0 else 0

    print(f"\n{'='*60}")
    print(f"  Résultats — {sequence.name} ({args.detector})")
    print(f"{'='*60}")
    print(f"  Frames traitées   : {len(sequence)}")
    print(f"  Détections totales: {total_detections}")
    print(f"  Tracks générés    : {total_tracks}")
    print(f"  Temps total       : {elapsed_total:.1f}s")
    print(f"  Vitesse moyenne   : {avg_fps:.2f} fps")
    print(f"  Résultat écrit    : {result_file}")
    print(f"{'='*60}\n")

    if video_writer is not None:
        video_writer.release()
    if args.visualize:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
