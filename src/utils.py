# src/utils.py
# Utilitaires pour la lecture des séquences MOT17
# et l'écriture des résultats au format MOTChallenge

import os
import cv2
import numpy as np
import configparser
from pathlib import Path
from typing import Iterator, Tuple


# ------------------------------------------------------------------
# Lecture d'une séquence MOT17
# ------------------------------------------------------------------

class MOT17Sequence:
    """
    Itérateur sur une séquence MOT17.

    Structure d'une séquence MOT17 :
        MOT17-02-DPM/
        ├── seqinfo.ini          ← métadonnées (fps, résolution, nb frames)
        ├── img1/                ← frames (000001.jpg, 000002.jpg, …)
        │   ├── 000001.jpg
        │   └── ...
        └── gt/
            └── gt.txt           ← annotations ground truth

    Format gt.txt (une ligne par objet) :
        frame, id, x, y, w, h, conf, class, visibility
    """

    def __init__(self, sequence_dir: str):
        self.seq_dir = Path(sequence_dir)
        self.name    = self.seq_dir.name

        # Lecture des métadonnées
        info_path = self.seq_dir / "seqinfo.ini"
        if not info_path.exists():
            raise FileNotFoundError(f"seqinfo.ini manquant dans {sequence_dir}")

        parser = configparser.ConfigParser()
        parser.read(str(info_path))
        seq = parser["Sequence"]

        self.seq_length  = int(seq["seqLength"])
        self.fps         = int(seq.get("frameRate", 30))
        self.img_width   = int(seq["imWidth"])
        self.img_height  = int(seq["imHeight"])
        self.img_dir     = self.seq_dir / seq.get("imDir", "img1")
        self.img_ext     = seq.get("imExt", ".jpg")

        print(
            f"[Sequence] {self.name} | "
            f"{self.seq_length} frames | "
            f"{self.img_width}×{self.img_height} | "
            f"{self.fps} fps"
        )

    def __len__(self) -> int:
        return self.seq_length

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Itère sur les frames de la séquence.
        Retourne (frame_id, frame_bgr) avec frame_id commençant à 1.
        """
        for frame_id in range(1, self.seq_length + 1):
            img_path = self.img_dir / f"{frame_id:06d}{self.img_ext}"
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"[WARNING] Frame manquante : {img_path}")
                continue
            yield frame_id, frame

    def load_ground_truth(self) -> dict:
        """
        Charge les annotations ground truth de la séquence.

        Retourne
        --------
        gt : dict
            {frame_id: np.ndarray shape (N, 6)}
            Chaque ligne : [id, x1, y1, x2, y2, visibility]
        """
        gt_path = self.seq_dir / "gt" / "gt.txt"
        if not gt_path.exists():
            print(f"[WARNING] Pas de ground truth pour {self.name}")
            return {}

        gt = {}
        with open(gt_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue

                frame_id  = int(parts[0])
                track_id  = int(parts[1])
                x         = float(parts[2])
                y         = float(parts[3])
                w         = float(parts[4])
                h         = float(parts[5])
                conf      = int(parts[6])    # 1 = à évaluer, 0 = à ignorer
                cls       = int(parts[7]) if len(parts) > 7 else 1
                visibility = float(parts[8]) if len(parts) > 8 else 1.0

                # Ignorer les annotations non évaluées (conf == 0)
                # et les classes non-piéton (cls != 1)
                if conf == 0 or cls != 1:
                    continue

                x2 = x + w
                y2 = y + h

                if frame_id not in gt:
                    gt[frame_id] = []
                gt[frame_id].append([track_id, x, y, x2, y2, visibility])

        return {fid: np.array(rows, dtype=np.float32)
                for fid, rows in gt.items()}


# ------------------------------------------------------------------
# Écriture des résultats au format MOTChallenge
# ------------------------------------------------------------------

class MOTResultWriter:
    """
    Écrit les résultats de tracking au format MOTChallenge.
    Format : <frame>,<id>,<x>,<y>,<w>,<h>,<conf>,-1,-1,-1
    Les coordonnées sont en pixels, w et h sont la largeur/hauteur.
    """

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "w")
        print(f"[Writer] Résultats → {self.output_path}")

    def write(self, frame_id: int, tracks: np.ndarray):
        """
        Écrit les pistes d'une frame dans le fichier résultat.

        Paramètres
        ----------
        frame_id : int
        tracks   : np.ndarray, shape (N, 5)
                   Chaque ligne : [x1, y1, x2, y2, track_id]
        """
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            w = x2 - x1
            h = y2 - y1
            # Conf = -1 signifie "non fourni" dans le format MOT
            self._file.write(
                f"{frame_id},{int(track_id)},"
                f"{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
                f"-1,-1,-1,-1\n"
            )

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ------------------------------------------------------------------
# Découverte des séquences MOT17
# ------------------------------------------------------------------

def list_mot17_sequences(data_dir: str, split: str = "train") -> list:
    """
    Liste les séquences MOT17 disponibles pour un split donné.

    Paramètres
    ----------
    data_dir : str
        Chemin vers data/MOT17/
    split : str
        "train" ou "test"

    Retourne
    --------
    sequences : list[Path]
        Chemins triés vers les dossiers de séquences.
    """
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Dossier MOT17 introuvable : {split_dir}\n"
            "Téléchargez MOT17 depuis https://motchallenge.net/data/MOT17/"
        )

    sequences = sorted([
        p for p in split_dir.iterdir()
        if p.is_dir() and p.name.startswith("MOT17-")
    ])

    print(f"[Dataset] MOT17/{split} : {len(sequences)} séquences trouvées")
    return sequences


# ------------------------------------------------------------------
# Visualisation (optionnelle, désactivable)
# ------------------------------------------------------------------

# Palette de couleurs pour les IDs (cycle sur 20 couleurs)
_PALETTE = [
    (255,  56,  56), (255, 157,  51), (255, 112,  31), (255, 178,  29),
    (207, 210,  49), (72,  249,  10), (146, 204,  23), ( 61, 219, 134),
    ( 26, 147,  52), (  0, 212, 187), ( 44, 153, 168), (  0, 194, 255),
    ( 52,  69, 147), (100,  45, 144), (142,  27, 100), (208,  11,  48),
    (255,   7,  58), (255, 122,  77), (245, 125,  20), (139, 125,  20),
]

def draw_tracks(frame_bgr: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    """
    Dessine les bounding boxes et IDs sur la frame.
    Utilisé uniquement pour la visualisation (non requis pour l'évaluation).
    """
    frame = frame_bgr.copy()
    for track in tracks:
        x1, y1, x2, y2, track_id = [int(v) for v in track]
        color = _PALETTE[int(track_id) % len(_PALETTE)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{track_id}"
        cv2.putText(
            frame, label, (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    return frame
