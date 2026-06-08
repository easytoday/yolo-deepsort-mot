# run_ablation.py
# Pilote de l'étude expérimentale — YOLOv3 / YOLOv8 + Deep SORT sur MOT17
# Gère l'exécution étalée sur plusieurs jours avec reprise automatique
# et le choix du détecteur (yolov3 ou yolov8) par expérience.
#
# Usage :
#   python run_ablation.py --status            # état d'avancement
#   python run_ablation.py --exp v8_baseline   # une expérience précise
#   python run_ablation.py --axis detector_v8  # tout un axe
#   python run_ablation.py --all               # tout ce qui reste
#   python run_ablation.py --report            # tableau comparatif final
#   python run_ablation.py --exp res_608 --force   # re-exécuter même si déjà fait

import sys
import os
import argparse
import time
import json
import yaml
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "deep_sort"))

# État persistant : quelles expériences sont terminées
STATE_FILE   = ROOT / "results" / "ablation" / "state.json"
EXPERIMENTS  = ROOT / "experiments.yaml"
BASE_TRK_CFG = ROOT / "configs" / "deepsort.yaml"


# ------------------------------------------------------------------
# Chargement config et état
# ------------------------------------------------------------------

def load_experiments():
    with open(EXPERIMENTS, "r") as f:
        return yaml.safe_load(f)


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ------------------------------------------------------------------
# Construction des configs surchargées (en mémoire)
# ------------------------------------------------------------------

def build_configs(exp):
    """
    Construit les dictionnaires de config détecteur + tracker pour une
    expérience. La config détecteur de base est choisie selon le type
    (yolov3.yaml ou yolov8.yaml), les surcharges de experiments.yaml
    sont appliquées ensuite.
    """
    detector_type = exp.get("detector_type", "yolov3")
    base_det = ROOT / "configs" / f"{detector_type}.yaml"

    with open(base_det, "r") as f:
        det_cfg = yaml.safe_load(f)
    with open(BASE_TRK_CFG, "r") as f:
        trk_cfg = yaml.safe_load(f)

    # Surcharges détecteur (img_size, conf_threshold, nms_threshold...)
    for key, val in exp.get("detector", {}).items():
        det_cfg["inference"][key] = val

    # Surcharges tracker (max_age, max_cosine_distance, n_init...)
    for key, val in exp.get("tracker", {}).items():
        trk_cfg["tracker"][key] = val

    return det_cfg, trk_cfg


# ------------------------------------------------------------------
# Exécution d'une expérience
# ------------------------------------------------------------------

def run_experiment(exp, sequences, force=False):
    """
    Exécute le tracking pour une expérience sur toutes les séquences,
    puis l'évaluation. Sauvegarde les résultats dans un dossier dédié.
    """
    from src.detector_factory import build_detector
    from src.tracker          import DeepSORTTracker
    from src.utils            import MOT17Sequence, MOTResultWriter

    exp_id        = exp["id"]
    detector_type = exp.get("detector_type", "yolov3")
    state         = load_state()

    if state.get(exp_id, {}).get("done") and not force:
        print(f"[SKIP] {exp_id} déjà terminée (--force pour relancer)")
        return

    print(f"\n{'='*60}")
    print(f"  EXPÉRIENCE : {exp_id}")
    print(f"  Détecteur  : {detector_type}")
    print(f"  Axe        : {exp['axis']}")
    print(f"  {exp['description']}")
    print(f"{'='*60}")

    # Construire les configs surchargées
    det_cfg, trk_cfg = build_configs(exp)

    out_dir = ROOT / "results" / "ablation" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tracer la config effective utilisée (reproductibilité)
    with open(out_dir / "config_used.yaml", "w") as f:
        yaml.dump({"detector_type": detector_type,
                   "detector": det_cfg["inference"],
                   "tracker": trk_cfg["tracker"]}, f, default_flow_style=False)

    # Écrire des fichiers de config temporaires pour instancier les classes
    tmp_det = out_dir / "_det_cfg.yaml"
    tmp_trk = out_dir / "_trk_cfg.yaml"
    with open(tmp_det, "w") as f:
        yaml.dump(det_cfg, f)
    with open(tmp_trk, "w") as f:
        yaml.dump(trk_cfg, f)

    # Charger le bon détecteur via la factory (yolov3 ou yolov8)
    print(f"\n[Init] Chargement détecteur {detector_type} "
          f"(img_size={det_cfg['inference']['img_size']}, "
          f"conf={det_cfg['inference']['conf_threshold']})...")
    detector = build_detector(detector_type, str(tmp_det))

    print(f"[Init] Chargement tracker (max_age={trk_cfg['tracker']['max_age']}, "
          f"cosine={trk_cfg['tracker']['max_cosine_distance']})...")
    tracker  = DeepSORTTracker(str(tmp_trk))

    exp_start = time.time()

    for seq_idx, seq_name in enumerate(sequences, 1):
        seq_path = ROOT / "data" / "MOT17" / "train" / seq_name
        if not seq_path.exists():
            print(f"  [WARN] Séquence absente : {seq_path}")
            continue

        print(f"\n  [{seq_idx}/{len(sequences)}] {seq_name}")
        tracker.reset()
        sequence    = MOT17Sequence(str(seq_path))
        result_file = out_dir / f"{seq_name}.txt"
        seq_start   = time.time()

        with MOTResultWriter(str(result_file)) as writer:
            for frame_id, frame_bgr in sequence:
                detections = detector.detect(frame_bgr)
                tracks     = tracker.update(detections, frame_bgr)
                writer.write(frame_id, tracks)
                if frame_id % 200 == 0:
                    elapsed = time.time() - seq_start
                    fps = frame_id / elapsed if elapsed > 0 else 0
                    print(f"    Frame {frame_id}/{len(sequence)} | {fps:.2f} fps")

        seq_elapsed = time.time() - seq_start
        print(f"    ✓ {seq_name} terminé en {seq_elapsed:.0f}s")

    exp_elapsed = time.time() - exp_start

    # Nettoyer les fichiers temporaires
    tmp_det.unlink(missing_ok=True)
    tmp_trk.unlink(missing_ok=True)

    # Marquer l'expérience comme terminée
    state = load_state()
    state[exp_id] = {
        "done": True,
        "axis": exp["axis"],
        "detector_type": detector_type,
        "description": exp["description"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": round(exp_elapsed, 1),
        "detector": exp.get("detector", {}),
        "tracker": exp.get("tracker", {}),
    }
    save_state(state)

    print(f"\n  ✓ Expérience {exp_id} terminée en {exp_elapsed/60:.1f} min")
    print(f"    Résultats : {out_dir}")

    # Évaluation immédiate
    print(f"\n[Eval] Évaluation de {exp_id}...")
    evaluate_experiment(exp_id, sequences)


# ------------------------------------------------------------------
# Évaluation d'une expérience
# ------------------------------------------------------------------

def evaluate_experiment(exp_id, sequences):
    """
    Calcule les métriques MOT pour une expérience et les sauvegarde.
    """
    import motmetrics as mm
    import pandas as pd

    out_dir = ROOT / "results" / "ablation" / exp_id

    def load_mot_file(filepath):
        cols = ["frame", "id", "x", "y", "w", "h", "conf", "d1", "d2", "d3"]
        df = pd.read_csv(filepath, header=None, names=cols)
        return df[["frame", "id", "x", "y", "w", "h"]]

    def load_gt_file(gt_path):
        cols = ["frame", "id", "x", "y", "w", "h", "conf", "class", "vis"]
        df = pd.read_csv(gt_path, header=None, names=cols)
        df = df[(df["conf"] == 1) & (df["class"] == 1)]
        return df[["frame", "id", "x", "y", "w", "h"]]

    accumulators = []
    names = []

    for seq_name in sequences:
        result_file = out_dir / f"{seq_name}.txt"
        gt_path = ROOT / "data" / "MOT17" / "train" / seq_name / "gt" / "gt.txt"
        if not result_file.exists() or not gt_path.exists():
            continue

        gt_df  = load_gt_file(str(gt_path))
        hyp_df = load_mot_file(str(result_file))

        acc = mm.MOTAccumulator(auto_id=True)
        all_frames = sorted(set(gt_df["frame"]) | set(hyp_df["frame"]))

        for frame_id in all_frames:
            gt_f  = gt_df[gt_df["frame"] == frame_id]
            hyp_f = hyp_df[hyp_df["frame"] == frame_id]
            gt_ids  = gt_f["id"].values.tolist()
            hyp_ids = hyp_f["id"].values.tolist()
            gt_boxes  = gt_f[["x", "y", "w", "h"]].values
            hyp_boxes = hyp_f[["x", "y", "w", "h"]].values
            dist = mm.distances.iou_matrix(gt_boxes, hyp_boxes, max_iou=0.5)
            acc.update(gt_ids, hyp_ids, dist)

        accumulators.append(acc)
        names.append(seq_name)

    if not accumulators:
        print("  [WARN] Aucun résultat à évaluer")
        return

    mh = mm.metrics.create()
    metrics = ["num_frames", "mota", "motp", "idf1", "num_switches",
               "mostly_tracked", "mostly_lost", "num_false_positives",
               "num_misses", "num_fragmentations"]
    summary = mh.compute_many(accumulators, metrics=metrics, names=names,
                              generate_overall=True)

    strsummary = mm.io.render_summary(
        summary, formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names)
    print(strsummary)

    summary.to_csv(str(out_dir / "metrics.csv"))
    print(f"  Métriques → {out_dir / 'metrics.csv'}")


# ------------------------------------------------------------------
# Affichage de l'état
# ------------------------------------------------------------------

def show_status(config):
    state = load_state()
    print(f"\n{'='*64}")
    print(f"  ÉTAT D'AVANCEMENT")
    print(f"{'='*64}")
    print(f"  {'Expérience':<14} {'Détect.':<8} {'Axe':<12} {'État':<10} {'Durée'}")
    print(f"  {'-'*14} {'-'*8} {'-'*12} {'-'*10} {'-'*10}")

    done_count = 0
    for exp in config["experiments"]:
        exp_id = exp["id"]
        det    = exp.get("detector_type", "yolov3")
        if state.get(exp_id, {}).get("done"):
            status = "✓ fait"
            dur    = f"{state[exp_id]['elapsed_s']/60:.0f} min"
            done_count += 1
        else:
            status = "à faire"
            dur    = "-"
        print(f"  {exp_id:<14} {det:<8} {exp['axis']:<12} {status:<10} {dur}")

    total = len(config["experiments"])
    print(f"  {'-'*14} {'-'*8} {'-'*12} {'-'*10} {'-'*10}")
    print(f"  Progression : {done_count}/{total} expériences terminées")
    print(f"{'='*64}\n")


# ------------------------------------------------------------------
# Rapport comparatif final
# ------------------------------------------------------------------

def generate_report(config):
    import pandas as pd

    print(f"\n{'='*86}")
    print(f"  TABLEAU COMPARATIF — ÉTUDE EXPÉRIMENTALE")
    print(f"{'='*86}")

    rows = []
    for exp in config["experiments"]:
        exp_id = exp["id"]
        metrics_file = ROOT / "results" / "ablation" / exp_id / "metrics.csv"
        if not metrics_file.exists():
            continue
        df = pd.read_csv(metrics_file, index_col=0)
        if "OVERALL" not in df.index:
            continue
        overall = df.loc["OVERALL"]
        rows.append({
            "exp": exp_id,
            "detecteur": exp.get("detector_type", "yolov3"),
            "axe": exp["axis"],
            "MOTA": overall.get("mota", float("nan")),
            "MOTP": overall.get("motp", float("nan")),
            "IDF1": overall.get("idf1", float("nan")),
            "IDs":  int(overall.get("num_switches", 0)),
            "FP":   int(overall.get("num_false_positives", 0)),
            "FN":   int(overall.get("num_misses", 0)),
        })

    if not rows:
        print("  Aucun résultat disponible. Lancez d'abord des expériences.")
        return

    print(f"\n  {'Expérience':<14} {'Détect.':<8} {'Axe':<12} {'MOTA':>7} {'MOTP':>7} "
          f"{'IDF1':>7} {'IDs':>5} {'FP':>6} {'FN':>7}")
    print(f"  {'-'*14} {'-'*8} {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*6} {'-'*7}")
    for r in rows:
        print(f"  {r['exp']:<14} {r['detecteur']:<8} {r['axe']:<12} "
              f"{r['MOTA']*100:>6.1f}% {r['MOTP']:>7.3f} "
              f"{r['IDF1']*100:>6.1f}% {r['IDs']:>5} {r['FP']:>6} {r['FN']:>7}")

    report_df = pd.DataFrame(rows)
    report_path = ROOT / "results" / "ablation" / "comparison.csv"
    report_df.to_csv(str(report_path), index=False)
    print(f"\n  Tableau sauvegardé : {report_path}")
    print(f"{'='*86}\n")


# ------------------------------------------------------------------
# Programme principal
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Étude expérimentale YOLOv3/YOLOv8 + Deep SORT")
    parser.add_argument("--status", action="store_true", help="Afficher l'avancement")
    parser.add_argument("--exp", help="Lancer une expérience par son id")
    parser.add_argument("--axis", help="Lancer toutes les expériences d'un axe")
    parser.add_argument("--all", action="store_true", help="Lancer tout ce qui reste")
    parser.add_argument("--report", action="store_true", help="Générer le tableau comparatif")
    parser.add_argument("--force", action="store_true", help="Relancer même si déjà fait")
    args = parser.parse_args()

    config    = load_experiments()
    sequences = config["sequences"]

    if args.status:
        show_status(config)
        return

    if args.report:
        generate_report(config)
        return

    # Sélection des expériences à lancer
    to_run = []
    if args.exp:
        to_run = [e for e in config["experiments"] if e["id"] == args.exp]
        if not to_run:
            print(f"[ERREUR] Expérience inconnue : {args.exp}")
            print(f"  Disponibles : {[e['id'] for e in config['experiments']]}")
            return
    elif args.axis:
        to_run = [e for e in config["experiments"] if e["axis"] == args.axis]
        if not to_run:
            print(f"[ERREUR] Axe inconnu : {args.axis}")
            print(f"  Disponibles : {sorted(set(e['axis'] for e in config['experiments']))}")
            return
    elif args.all:
        to_run = config["experiments"]
    else:
        print("Aucune action spécifiée. Options :")
        print("  --status | --exp ID | --axis AXE | --all | --report")
        show_status(config)
        return

    print(f"\n[Ablation] {len(to_run)} expérience(s) à traiter")
    for exp in to_run:
        run_experiment(exp, sequences, force=args.force)

    print(f"\n[Ablation] Terminé. Voir le tableau : python run_ablation.py --report")


if __name__ == "__main__":
    main()
