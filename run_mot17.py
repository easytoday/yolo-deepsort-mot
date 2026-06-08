# run_mot17.py
# Script batch : traitement de toutes les séquences MOT17 (train, pas de test)
# Usage : python run_mot17.py --split train
#         python run_mot17.py --split train --detector yolov8
#         python run_mot17.py --split train --sequences MOT17-02 MOT17-04

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "deep_sort"))

from src.detector_factory import build_detector
from src.tracker          import DeepSORTTracker
from src.utils            import MOT17Sequence, MOTResultWriter, list_mot17_sequences


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv3/YOLOv8 + Deep SORT — Traitement batch MOT17"
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Split MOT17 à traiter (défaut: train)"
    )
    parser.add_argument(
        "--detector",
        choices=["yolov3", "yolov8"],
        default="yolov3",
        help="Détecteur à utiliser (défaut: yolov3)"
    )
    parser.add_argument(
        "--data-dir",
        default="data/MOT17",
        help="Dossier racine MOT17 (défaut: data/MOT17)"
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Sous-ensemble de séquences (ex: MOT17-02 MOT17-04). "
             "Si absent, toutes les séquences du split sont traitées."
    )
    parser.add_argument(
        "--detector-config",
        default=None,
        help="Fichier de config détecteur. Si absent, défaut du type choisi."
    )
    parser.add_argument(
        "--tracker-config",
        default="configs/deepsort.yaml"
    )
    parser.add_argument(
        "--output-dir",
        default="results/mot17"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Découverte des séquences
    # ------------------------------------------------------------------
    all_sequences = list_mot17_sequences(args.data_dir, args.split)

    # Filtrage si l'utilisateur a spécifié un sous-ensemble
    if args.sequences:
        all_sequences = [
            s for s in all_sequences
            if any(name in s.name for name in args.sequences)
        ]
        print(f"[Batch] Sous-ensemble sélectionné : {[s.name for s in all_sequences]}")

    if not all_sequences:
        print("[ERREUR] Aucune séquence trouvée. Vérifiez --data-dir et --sequences.")
        return

    # ------------------------------------------------------------------
    # 2. Chargement unique du détecteur et du tracker
    # ------------------------------------------------------------------
    # La factory choisit YOLOv3 ou YOLOv8 selon --detector.
    print(f"[Batch] Détecteur : {args.detector}")
    detector = build_detector(args.detector, args.detector_config)
    tracker  = DeepSORTTracker(args.tracker_config)

    # ------------------------------------------------------------------
    # 3. Traitement séquence par séquence
    # ------------------------------------------------------------------
    print(f"\n[Batch] {len(all_sequences)} séquence(s) à traiter\n")
    global_start = time.time()
    results_summary = []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for seq_idx, seq_path in enumerate(all_sequences, 1):
        print(f"\n[{seq_idx}/{len(all_sequences)}] {seq_path.name}")
        print("-" * 50)

        # Réinitialiser le tracker entre les séquences (IDs repartent à 1)
        tracker.reset()

        sequence    = MOT17Sequence(str(seq_path))
        result_file = output_dir / f"{sequence.name}.txt"
        seq_start   = time.time()
        n_frames    = 0

        with MOTResultWriter(str(result_file)) as writer:
            for frame_id, frame_bgr in sequence:
                detections = detector.detect(frame_bgr)
                tracks     = tracker.update(detections, frame_bgr)
                writer.write(frame_id, tracks)
                n_frames += 1

                # Progression allégée pour le batch
                if frame_id % 100 == 0:
                    elapsed = time.time() - seq_start
                    fps     = frame_id / elapsed if elapsed > 0 else 0
                    print(f"    Frame {frame_id}/{len(sequence)} | {fps:.2f} fps")

        seq_elapsed = time.time() - seq_start
        seq_fps     = n_frames / seq_elapsed if seq_elapsed > 0 else 0

        results_summary.append({
            "sequence": sequence.name,
            "frames"  : n_frames,
            "time_s"  : seq_elapsed,
            "fps"     : seq_fps,
            "output"  : str(result_file),
        })

        print(f"  Terminé en {seq_elapsed:.1f}s ({seq_fps:.2f} fps) → {result_file}")

    # ------------------------------------------------------------------
    # 4. Résumé global
    # ------------------------------------------------------------------
    global_elapsed = time.time() - global_start
    total_frames   = sum(r["frames"] for r in results_summary)

    print(f"\n{'='*60}")
    print(f"  RÉSUMÉ BATCH — détecteur {args.detector}")
    print(f"{'='*60}")
    print(f"  {'Séquence':<25} {'Frames':>7} {'Temps':>8} {'FPS':>6}")
    print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*6}")
    for r in results_summary:
        print(
            f"  {r['sequence']:<25} {r['frames']:>7} "
            f"{r['time_s']:>7.1f}s {r['fps']:>5.2f}"
        )
    print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*6}")
    print(
        f"  {'TOTAL':<25} {total_frames:>7} "
        f"{global_elapsed:>7.1f}s "
        f"{total_frames/global_elapsed:>5.2f}"
    )
    print(f"\n  Résultats dans : {output_dir}/")
    print(f"  Prochaine étape : python evaluate.py --split {args.split}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
