# evaluate.py
# Calcul des métriques MOT (MOTA, MOTP, IDF1, etc.)
# Utilise la librairie motmetrics (pip) sur les fichiers résultats
# Usage : python evaluate.py --split train
#         python evaluate.py --split train --sequence MOT17-02-DPM

import sys
import argparse
import os
import numpy as np
import pandas as pd
import motmetrics as mm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import MOT17Sequence, list_mot17_sequences


def parse_args():
    parser = argparse.ArgumentParser(
        description="Évaluation MOT17 — MOTA, MOTP, IDF1, MT, ML"
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Split MOT17 (défaut: train)"
    )
    parser.add_argument(
        "--data-dir",
        default="data/MOT17",
        help="Dossier racine MOT17 (défaut: data/MOT17)"
    )
    parser.add_argument(
        "--results-dir",
        default="results/mot17",
        help="Dossier contenant les fichiers .txt résultats"
    )
    parser.add_argument(
        "--sequence",
        default=None,
        help="Évaluer une seule séquence (ex: MOT17-02-DPM)"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Seuil IoU pour l'association GT ↔ hypothèse (défaut: 0.5)"
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Lecture des fichiers de résultats au format MOT
# ------------------------------------------------------------------

def load_mot_file(filepath: str) -> pd.DataFrame:
    """
    Charge un fichier de résultats au format MOTChallenge.
    Format : frame,id,x,y,w,h,conf,-1,-1,-1

    Retourne un DataFrame avec colonnes :
        frame, id, x, y, w, h, conf
    """
    cols = ["frame", "id", "x", "y", "w", "h", "conf",
            "dummy1", "dummy2", "dummy3"]
    df = pd.read_csv(filepath, header=None, names=cols)
    return df[["frame", "id", "x", "y", "w", "h", "conf"]]


def load_gt_file(gt_path: str) -> pd.DataFrame:
    """
    Charge le fichier ground truth MOT17.
    Format : frame,id,x,y,w,h,conf,class,visibility

    Retourne un DataFrame filtré sur les piétons (class=1, conf=1).
    """
    cols = ["frame", "id", "x", "y", "w", "h", "conf", "class", "visibility"]
    df = pd.read_csv(gt_path, header=None, names=cols)

    # Filtrer : uniquement piétons annotés (conf=1, class=1)
    df = df[(df["conf"] == 1) & (df["class"] == 1)]
    return df[["frame", "id", "x", "y", "w", "h", "visibility"]]


# ------------------------------------------------------------------
# Calcul de l'IoU entre boîtes
# ------------------------------------------------------------------

def iou_matrix(gt_boxes: np.ndarray, hyp_boxes: np.ndarray) -> np.ndarray:
    """
    Calcule la matrice IoU entre les boîtes GT et les hypothèses.
    motmetrics attend une matrice de distances = 1 - IoU.

    Paramètres
    ----------
    gt_boxes  : (N, 4) — [x, y, w, h]
    hyp_boxes : (M, 4) — [x, y, w, h]

    Retourne
    --------
    dist_matrix : (N, M) — distances = 1 - IoU
    """
    if len(gt_boxes) == 0 or len(hyp_boxes) == 0:
        return np.empty((len(gt_boxes), len(hyp_boxes)))

    # Conversion [x, y, w, h] → [x1, y1, x2, y2]
    gt_xyxy  = np.stack([
        gt_boxes[:, 0],
        gt_boxes[:, 1],
        gt_boxes[:, 0] + gt_boxes[:, 2],
        gt_boxes[:, 1] + gt_boxes[:, 3],
    ], axis=1)

    hyp_xyxy = np.stack([
        hyp_boxes[:, 0],
        hyp_boxes[:, 1],
        hyp_boxes[:, 0] + hyp_boxes[:, 2],
        hyp_boxes[:, 1] + hyp_boxes[:, 3],
    ], axis=1)

    # Calcul vectorisé des IoU
    dist = np.zeros((len(gt_xyxy), len(hyp_xyxy)))
    for i, g in enumerate(gt_xyxy):
        xi1 = np.maximum(g[0], hyp_xyxy[:, 0])
        yi1 = np.maximum(g[1], hyp_xyxy[:, 1])
        xi2 = np.minimum(g[2], hyp_xyxy[:, 2])
        yi2 = np.minimum(g[3], hyp_xyxy[:, 3])

        inter = np.maximum(0, xi2 - xi1) * np.maximum(0, yi2 - yi1)
        area_g   = (g[2] - g[0]) * (g[3] - g[1])
        area_hyp = (hyp_xyxy[:, 2] - hyp_xyxy[:, 0]) * (hyp_xyxy[:, 3] - hyp_xyxy[:, 1])
        union = area_g + area_hyp - inter
        iou   = np.where(union > 0, inter / union, 0.0)
        dist[i] = 1.0 - iou

    return dist


# ------------------------------------------------------------------
# Évaluation d'une séquence
# ------------------------------------------------------------------

def evaluate_sequence(
    seq_name: str,
    gt_df: pd.DataFrame,
    hyp_df: pd.DataFrame,
    iou_threshold: float,
) -> mm.MOTAccumulator:
    """
    Évalue une séquence en construisant un accumulateur motmetrics.

    Paramètres
    ----------
    seq_name      : nom de la séquence (pour logs)
    gt_df         : DataFrame ground truth
    hyp_df        : DataFrame hypothèses (résultats tracker)
    iou_threshold : seuil IoU pour considérer une association correcte

    Retourne
    --------
    acc : mm.MOTAccumulator
    """
    acc = mm.MOTAccumulator(auto_id=True)

    all_frames = sorted(set(gt_df["frame"].unique()) | set(hyp_df["frame"].unique()))

    for frame_id in all_frames:
        # Ground truth pour cette frame
        gt_frame = gt_df[gt_df["frame"] == frame_id]
        gt_ids   = gt_frame["id"].values.tolist()
        gt_boxes = gt_frame[["x", "y", "w", "h"]].values

        # Hypothèses pour cette frame
        hyp_frame = hyp_df[hyp_df["frame"] == frame_id]
        hyp_ids   = hyp_frame["id"].values.tolist()
        hyp_boxes = hyp_frame[["x", "y", "w", "h"]].values

        # Matrice de distances (1 - IoU)
        if len(gt_ids) > 0 and len(hyp_ids) > 0:
            dist = iou_matrix(gt_boxes, hyp_boxes)
            # Masquer les associations dont l'IoU est inférieur au seuil
            dist[dist > (1.0 - iou_threshold)] = np.nan
        else:
            dist = mm.distances.iou_matrix(
                gt_boxes, hyp_boxes, max_iou=iou_threshold
            ) if len(gt_ids) > 0 or len(hyp_ids) > 0 else np.empty((0, 0))

        acc.update(gt_ids, hyp_ids, dist)

    return acc


# ------------------------------------------------------------------
# Programme principal
# ------------------------------------------------------------------

def main():
    args = parse_args()

    # Découverte des séquences à évaluer
    all_seq_paths = list_mot17_sequences(args.data_dir, args.split)

    if args.sequence:
        all_seq_paths = [s for s in all_seq_paths if args.sequence in s.name]

    if not all_seq_paths:
        print("[ERREUR] Aucune séquence à évaluer.")
        return

    results_dir = Path(args.results_dir)
    accumulators = []
    seq_names    = []

    print(f"\n[Eval] Évaluation sur {len(all_seq_paths)} séquence(s)")
    print(f"       IoU threshold : {args.iou_threshold}\n")

    for seq_path in all_seq_paths:
        seq_name = seq_path.name
        gt_path  = seq_path / "gt" / "gt.txt"

        # Chercher le fichier de résultats
        # Note : MOT17 a 3 détecteurs (DPM, FRCNN, SDP) par séquence
        # On cherche le fichier correspondant au nom exact
        result_file = results_dir / f"{seq_name}.txt"

        if not result_file.exists():
            print(f"  [SKIP] Résultat manquant : {result_file}")
            continue
        if not gt_path.exists():
            print(f"  [SKIP] GT manquant : {gt_path}")
            continue

        print(f"  Évaluation de {seq_name}...")
        gt_df  = load_gt_file(str(gt_path))
        hyp_df = load_mot_file(str(result_file))

        acc = evaluate_sequence(seq_name, gt_df, hyp_df, args.iou_threshold)
        accumulators.append(acc)
        seq_names.append(seq_name)

    if not accumulators:
        print("[ERREUR] Aucun accumulateur — vérifiez les chemins.")
        return

    # ------------------------------------------------------------------
    # Calcul des métriques avec motmetrics
    # ------------------------------------------------------------------
    mh = mm.metrics.create()

    # Métriques à calculer (standard MOT benchmark)
    metrics = [
        "num_frames",
        "mota",          # Multi-Object Tracking Accuracy
        "motp",          # Multi-Object Tracking Precision
        "idf1",          # Identification F1
        "num_switches",  # ID Switches
        "mostly_tracked",
        "mostly_lost",
        "num_false_positives",
        "num_misses",
        "num_fragmentations",
    ]

    summary = mh.compute_many(
        accumulators,
        metrics=metrics,
        names=seq_names,
        generate_overall=True,
    )

    # Formatage de l'affichage
    formatters = mm.io.motchallenge_metric_names
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )

    print("\n" + "="*80)
    print("  RÉSULTATS — MOT17 / YOLOv3 + Deep SORT")
    print("="*80)
    print(strsummary)
    print("="*80)

    # Sauvegarde CSV
    csv_path = results_dir / f"metrics_{args.split}.csv"
    summary.to_csv(str(csv_path))
    print(f"\n  Métriques sauvegardées : {csv_path}")
    print(f"\n  Pour comparer avec le leaderboard :")
    print(f"  → https://motchallenge.net/results/MOT17/\n")


if __name__ == "__main__":
    main()
