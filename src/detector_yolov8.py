# src/detector_yolov8.py
# Wrapper YOLOv8 pour la détection d'objets par frame
# Backend : package ultralytics (PyTorch, CPU)
#
# Interface identique à YOLOv3Detector :
#   detector = YOLOv8Detector("configs/yolov8.yaml")
#   detections = detector.detect(frame_bgr)
#   # → np.ndarray shape (N, 5) : [x1, y1, x2, y2, confidence]
#
# Cela garantit que Deep SORT, l'évaluation et tout
# le reste du pipeline fonctionnent sans aucune modification.

import os
import numpy as np
import yaml


class YOLOv8Detector:
    """
    Wrapper YOLOv8 via le package ultralytics.

    Entrée  : frame BGR (np.ndarray H×W×3)
    Sortie  : détections np.ndarray (N, 5) → [x1, y1, x2, y2, confidence]
              coordonnées absolues dans l'image originale
    """

    def __init__(self, config_path: str):
        """
        Paramètres
        ----------
        config_path : str
            Chemin vers configs/yolov8.yaml
        """
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.img_size    = self.cfg["inference"]["img_size"]
        self.conf_thresh = self.cfg["inference"]["conf_threshold"]
        self.nms_thresh  = self.cfg["inference"]["nms_threshold"]
        self.target_cls  = set(self.cfg["inference"]["target_classes"])

        self._load_model()

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------

    def _load_model(self):
        """
        Charge le modèle YOLOv8 via ultralytics.
        Le fichier de poids (.pt) est téléchargé automatiquement par
        ultralytics au premier usage s'il n'est pas présent localement.
        """
        weights_path = self.cfg["model"]["weights"]

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Package ultralytics manquant.\n"
                "Installez-le : pip install ultralytics\n"
                "(ou ajoutez-le à environment.yml puis conda env update)"
            )

        # Le modèle se charge depuis un .pt local ou se télécharge
        # automatiquement (ex: 'yolov8s.pt')
        self.model = YOLO(weights_path)

        # Forcer l'exécution sur CPU
        self.device = "cpu"

        print(
            f"[Detector] YOLOv8 chargé via ultralytics (CPU)\n"
            f"           poids    : {weights_path}\n"
            f"           img_size : {self.img_size}px\n"
            f"           conf     : {self.conf_thresh}"
        )

    # ------------------------------------------------------------------
    # Détection
    # ------------------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Détecte les objets cibles dans une frame.

        Paramètres
        ----------
        frame_bgr : np.ndarray
            Frame au format BGR (sortie OpenCV), shape (H, W, 3)

        Retourne
        --------
        detections : np.ndarray, shape (N, 5)
            Chaque ligne : [x1, y1, x2, y2, confidence]
            Coordonnées absolues (pixels) dans l'image originale.
            Tableau vide (0, 5) si aucune détection.
        """
        # ultralytics accepte directement une image BGR (numpy)
        # et gère le redimensionnement / la NMS en interne
        results = self.model.predict(
            source=frame_bgr,
            imgsz=self.img_size,
            conf=self.conf_thresh,
            iou=self.nms_thresh,
            classes=list(self.target_cls),   # filtrage classe (0 = personne)
            device=self.device,
            verbose=False,                    # pas de log par frame
        )

        # results est une liste (une entrée par image) ; on traite la 1ère
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)

        # Extraction des boîtes au format [x1, y1, x2, y2] et confiances
        # .xyxy retourne déjà les coordonnées absolues dans l'image originale
        boxes = result.boxes.xyxy.cpu().numpy()   # shape (N, 4)
        confs = result.boxes.conf.cpu().numpy()   # shape (N,)

        detections = np.column_stack([boxes, confs]).astype(np.float32)
        return detections
