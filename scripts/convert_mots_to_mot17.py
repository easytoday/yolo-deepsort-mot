# scripts/convert_mots_to_mot17.py
# Convertit la structure MOTSChallenge → structure MOT17 standard
# attendue par run_tracker.py, run_mot17.py et evaluate.py
#
# Nécessite : pip install pycocotools
#
# Usage : python scripts/convert_mots_to_mot17.py
#         python scripts/convert_mots_to_mot17.py --data-dir data/MOT17

import sys
import os
import argparse
import configparser
from pathlib import Path
import numpy as np


# ------------------------------------------------------------------
# Vérification pycocotools
# ------------------------------------------------------------------
try:
    from pycocotools import mask as cocomask
except ImportError:
    print("[ERREUR] pycocotools manquant.")
    print("  Installez-le : pip install pycocotools")
    sys.exit(1)


# ------------------------------------------------------------------
# Correspondance numéro MOTS → métadonnées MOT17
# ------------------------------------------------------------------
SEQ_MAP = {
    "0002": {"name": "MOT17-02-DPM", "fps": 30},
    "0005": {"name": "MOT17-05-DPM", "fps": 14},
    "0009": {"name": "MOT17-09-DPM", "fps": 30},
    "0011": {"name": "MOT17-11-DPM", "fps": 30},
}

# Dans MOTS : class_id=2 → piéton, class_id=1 → voiture
PEDESTRIAN_CLASS = 2
# id=10000 → région à ignorer
IGNORE_ID = 10000


# ------------------------------------------------------------------
# Décodage RLE → bounding box via pycocotools
# ------------------------------------------------------------------

def rle_str_to_bbox(rle_str: str, height: int, width: int):
    """
    Décode un masque RLE COCO (format compressé) et retourne
    sa bounding box [x, y, w, h] en pixels.

    Utilise pycocotools.mask pour un décodage correct.
    Retourne None si le masque est vide.
    """
    # pycocotools attend un dict {"counts": bytes, "size": [h, w]}
    rle = {
        "counts": rle_str.encode("utf-8"),
        "size":   [height, width],
    }

    # Décode le masque binaire (height × width)
    mask = cocomask.decode(rle)   # np.ndarray uint8, shape (H, W)

    if mask.sum() == 0:
        return None

    # Calcul de la bounding box à partir du masque
    # toBbox retourne [x, y, w, h] en float
    bbox = cocomask.toBbox(rle)   # [x, y, w, h]
    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

    if w <= 0 or h <= 0:
        return None

    return x, y, w, h


# ------------------------------------------------------------------
# Lecture des annotations MOTS txt → gt.txt MOT
# ------------------------------------------------------------------

def read_mots_txt(txt_path: str, height: int, width: int) -> dict:
    """
    Lit un fichier annotations MOTS (instances_txt/XXXX.txt).
    Format : time_frame id class_id img_height img_width rle

    Retourne dict {frame_id: [(track_id, x, y, w, h), ...]}
    uniquement pour les piétons.
    """
    gt         = {}
    n_total    = 0
    n_ped      = 0
    n_ok       = 0
    n_empty    = 0

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            frame_id = int(parts[0])
            obj_id   = int(parts[1])
            class_id = int(parts[2])
            img_h    = int(parts[3])
            img_w    = int(parts[4])
            rle_str  = parts[5]       # un seul token (pas d'espaces dans le RLE compressé)

            n_total += 1

            # Ignorer les régions à ignorer et les non-piétons
            if obj_id == IGNORE_ID or class_id != PEDESTRIAN_CLASS:
                continue

            n_ped += 1

            # instance_id encodé dans obj_id (class_id * 1000 + instance_id)
            instance_id = obj_id % 1000

            # Décodage RLE → bounding box
            bbox = rle_str_to_bbox(rle_str, img_h, img_w)
            if bbox is None:
                n_empty += 1
                continue

            x, y, w, h = bbox
            n_ok += 1

            if frame_id not in gt:
                gt[frame_id] = []
            gt[frame_id].append((instance_id, x, y, w, h))

    print(f"    Lignes : {n_total} total | {n_ped} piétons | "
          f"{n_ok} bbox valides | {n_empty} masques vides")
    return gt


# ------------------------------------------------------------------
# Écriture gt.txt format MOT standard
# ------------------------------------------------------------------

def write_mot_gt(gt: dict, output_path: Path):
    """
    Écrit les bounding boxes au format MOTChallenge :
    frame,id,x,y,w,h,conf,class,visibility
    conf=1 (annotation valide), class=1 (piéton MOT), visibility=1
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_lines = 0
    with open(output_path, "w") as f:
        for frame_id in sorted(gt.keys()):
            for (track_id, x, y, w, h) in gt[frame_id]:
                f.write(f"{frame_id},{track_id},"
                        f"{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                        f"1,1,1\n")
                n_lines += 1
    print(f"    gt.txt : {n_lines} lignes → {output_path}")


# ------------------------------------------------------------------
# Création du seqinfo.ini
# ------------------------------------------------------------------

def write_seqinfo(seq_dir: Path, seq_name: str, fps: int,
                  width: int, height: int, n_frames: int):
    ini_path = seq_dir / "seqinfo.ini"
    cfg = configparser.ConfigParser()
    cfg["Sequence"] = {
        "name":      seq_name,
        "imDir":     "img1",
        "frameRate": str(fps),
        "seqLength": str(n_frames),
        "imWidth":   str(width),
        "imHeight":  str(height),
        "imExt":     ".jpg",
    }
    with open(ini_path, "w") as f:
        cfg.write(f)
    print(f"    seqinfo.ini : {n_frames} frames {width}×{height} @{fps}fps")


# ------------------------------------------------------------------
# Conversion d'une séquence
# ------------------------------------------------------------------

def convert_sequence(seq_id: str, train_dir: Path, force: bool = False):
    if seq_id not in SEQ_MAP:
        print(f"  [SKIP] Séquence inconnue : {seq_id}")
        return

    info     = SEQ_MAP[seq_id]
    seq_name = info["name"]
    fps      = info["fps"]

    print(f"\n  [{seq_id}] → {seq_name}")

    src_images = train_dir / "images" / seq_id
    src_txt    = train_dir / "instances_txt" / f"{seq_id}.txt"
    dst_dir    = train_dir / seq_name
    dst_img1   = dst_dir / "img1"
    dst_gt     = dst_dir / "gt"

    if not src_images.exists():
        print(f"    [ERREUR] Images absentes : {src_images}")
        return

    dst_dir.mkdir(exist_ok=True)
    dst_img1.mkdir(exist_ok=True)
    dst_gt.mkdir(exist_ok=True)

    # 1. Symlinks vers les frames
    frames = sorted(list(src_images.glob("*.jpg")) +
                    list(src_images.glob("*.png")))
    n_frames = len(frames)
    n_links  = 0
    for frame_path in frames:
        link = dst_img1 / frame_path.name
        if link.exists() and not force:
            continue
        if link.exists():
            link.unlink()
        link.symlink_to(frame_path.resolve())
        n_links += 1
    print(f"    Frames : {n_frames} | Symlinks créés : {n_links}")

    # 2. Résolution réelle depuis la première frame
    width, height = 1920, 1080  # valeurs par défaut
    if frames:
        try:
            import cv2
            img = cv2.imread(str(frames[0]))
            if img is not None:
                height, width = img.shape[:2]
        except Exception:
            pass

    # 3. seqinfo.ini
    write_seqinfo(dst_dir, seq_name, fps, width, height, n_frames)

    # 4. Conversion annotations
    gt_path = dst_gt / "gt.txt"
    if gt_path.exists() and not force:
        print(f"    gt.txt déjà présent (--force pour recréer)")
    elif src_txt.exists():
        print(f"    Conversion RLE → bounding boxes (pycocotools)...")
        gt = read_mots_txt(str(src_txt), height, width)
        write_mot_gt(gt, gt_path)

        # Vérification rapide : afficher les 2 premières lignes du gt.txt
        with open(gt_path) as f:
            lines = [next(f) for _ in range(min(2, sum(1 for _ in open(gt_path))))]
        print(f"    Aperçu gt.txt : {lines[0].strip()}")
    else:
        print(f"    [AVERT] Annotations absentes : {src_txt}")


# ------------------------------------------------------------------
# Programme principal
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Conversion MOTSChallenge → MOT17 standard (pycocotools)"
    )
    parser.add_argument("--data-dir", default="data/MOT17")
    parser.add_argument("--force", action="store_true",
                        help="Recréer les fichiers même s'ils existent")
    args = parser.parse_args()

    train_dir = Path(args.data_dir) / "train"
    if not (train_dir / "images").exists():
        print(f"[ERREUR] {train_dir}/images/ absent")
        sys.exit(1)

    seq_ids = sorted([d.name for d in (train_dir / "images").iterdir()
                      if d.is_dir()])
    print(f"\n[Conversion] MOTSChallenge → MOT17")
    print(f"  Séquences : {seq_ids}")
    print(f"  Décodeur  : pycocotools (RLE officiel COCO)")

    for seq_id in seq_ids:
        convert_sequence(seq_id, train_dir, force=args.force)

    # Résumé
    print(f"\n{'='*55}")
    ok_count = 0
    for seq_id in seq_ids:
        if seq_id not in SEQ_MAP:
            continue
        seq_dir  = train_dir / SEQ_MAP[seq_id]["name"]
        n_frames = len(list((seq_dir / "img1").glob("*.jpg"))) if (seq_dir / "img1").exists() else 0
        gt_ok    = (seq_dir / "gt" / "gt.txt").exists()
        ini_ok   = (seq_dir / "seqinfo.ini").exists()
        ok       = n_frames > 0 and gt_ok and ini_ok
        if ok:
            ok_count += 1
        status = "✓" if ok else "✗"
        print(f"  {status} {SEQ_MAP[seq_id]['name']:20s} | "
              f"{n_frames:4d} frames | "
              f"gt={'OK' if gt_ok else 'MANQUANT':8s} | "
              f"ini={'OK' if ini_ok else 'MANQUANT'}")

    print(f"\n  {ok_count}/{len(seq_ids)} séquences prêtes")
    if ok_count == len(seq_ids):
        print(f"\n  Lancer le pipeline :")
        print(f"    python run_mot17.py --split train")
        print(f"    python evaluate.py  --split train")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
