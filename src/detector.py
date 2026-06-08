# src/detector.py
# Wrapper YOLOv3 pour la détection d'objets par frame
# Backend : OpenCV DNN (cv2.dnn.readNetFromDarknet)
#   → Charge directement les fichiers .cfg et .weights de darknet
#   → Pas de dépendance PyTorch pour l'inférence
#   → Plus performant que PyTorch sur CPU (backend OpenBLAS/OpenCV optimisé)
#
# Interface publique :
#   detector = YOLOv3Detector("configs/yolov3.yaml")
#   detections = detector.detect(frame_bgr)
#   # → np.ndarray shape (N, 5) : [x1, y1, x2, y2, confidence]

import os
import numpy as np
import cv2
import yaml


class YOLOv3Detector:
    """
    Wrapper YOLOv3 via OpenCV DNN.

    Entrée  : frame BGR (np.ndarray H×W×3)
    Sortie  : détections np.ndarray (N, 5) → [x1, y1, x2, y2, confidence]
              coordonnées absolues dans l'image originale
    """

    def __init__(self, config_path: str):
        """
        Paramètres
        ----------
        config_path : str
            Chemin vers configs/yolov3.yaml
        """
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.img_size    = self.cfg["inference"]["img_size"]
        self.conf_thresh = self.cfg["inference"]["conf_threshold"]
        self.nms_thresh  = self.cfg["inference"]["nms_threshold"]
        self.target_cls  = set(self.cfg["inference"]["target_classes"])

        self._load_class_names()
        self._load_network()

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------

    def _load_class_names(self):
        """Charge les noms de classes COCO depuis coco.names."""
        names_path = self.cfg["model"]["names"]
        if not os.path.exists(names_path):
            raise FileNotFoundError(
                f"Fichier classes introuvable : {names_path}\n"
                "Lancez d'abord : bash scripts/setup.sh"
            )
        with open(names_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]
        print(f"[Detector] {len(self.class_names)} classes COCO chargées")

    def _load_network(self):
        """
        Charge le réseau YOLOv3 via OpenCV DNN.
        cv2.dnn.readNetFromDarknet accepte directement les fichiers
        .cfg et .weights du format darknet original.
        """
        cfg_path     = self.cfg["model"]["cfg"]
        weights_path = self.cfg["model"]["weights"]

        for path in (cfg_path, weights_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Fichier YOLOv3 introuvable : {path}\n"
                    "Lancez d'abord : bash scripts/setup.sh"
                )

        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

        # Forcer CPU : pas d'accélération CUDA
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Noms des couches de sortie YOLOv3 (les 3 têtes de détection)
        layer_names = self.net.getLayerNames()
        out_indices = self.net.getUnconnectedOutLayers()
        # Compatibilité OpenCV 4.x (flat array) et anciennes versions (nested)
        if isinstance(out_indices[0], (list, np.ndarray)):
            out_indices = [i[0] for i in out_indices]
        self.output_layers = [layer_names[i - 1] for i in out_indices]

        print(
            f"[Detector] YOLOv3 chargé via OpenCV DNN (CPU)\n"
            f"           cfg     : {cfg_path}\n"
            f"           weights : {weights_path}\n"
            f"           img_size: {self.img_size}px\n"
            f"           sorties : {self.output_layers}"
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
        h, w = frame_bgr.shape[:2]

        # Préparation du blob d'entrée
        # blobFromImage : resize, normalise [0,1], BGR→RGB, pas de soustraction moyenne
        blob = cv2.dnn.blobFromImage(
            frame_bgr,
            scalefactor=1.0 / 255.0,
            size=(self.img_size, self.img_size),
            swapRB=True,    # BGR → RGB
            crop=False,
        )
        self.net.setInput(blob)

        # Inférence sur les 3 têtes YOLOv3
        raw_outputs = self.net.forward(self.output_layers)
        # raw_outputs : liste de 3 arrays, chacun shape (N_anchors, 5 + n_classes)
        # Colonnes : [cx, cy, bw, bh, obj_conf, cls0, cls1, ..., cls79]

        boxes       = []
        confidences = []

        for output in raw_outputs:
            for detection in output:
                scores     = detection[5:]          # scores par classe
                class_id   = int(np.argmax(scores))
                class_conf = float(scores[class_id])
                obj_conf   = float(detection[4])

                # Confiance finale = objectness × confiance de classe
                confidence = obj_conf * class_conf

                if confidence < self.conf_thresh:
                    continue
                if class_id not in self.target_cls:
                    continue

                # Conversion centre→coin (coordonnées relatives → absolues)
                cx = detection[0] * w
                cy = detection[1] * h
                bw = detection[2] * w
                bh = detection[3] * h

                x1 = cx - bw / 2
                y1 = cy - bh / 2
                x2 = cx + bw / 2
                y2 = cy + bh / 2

                # Clamp dans l'image
                x1 = max(0.0, min(x1, w))
                y1 = max(0.0, min(y1, h))
                x2 = max(0.0, min(x2, w))
                y2 = max(0.0, min(y2, h))

                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(confidence)

        if len(boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)

        # Non-Maximum Suppression via OpenCV
        boxes_xywh = [
            [b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes
        ]
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            confidences,
            score_threshold=self.conf_thresh,
            nms_threshold=self.nms_thresh,
        )

        if len(indices) == 0:
            return np.empty((0, 5), dtype=np.float32)

        # Aplatir les indices (format différent selon les versions d'OpenCV)
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        else:
            indices = [i[0] if isinstance(i, (list, np.ndarray)) else i
                       for i in indices]

        detections = np.array(
            [[*boxes[i], confidences[i]] for i in indices],
            dtype=np.float32,
        )
        return detections
