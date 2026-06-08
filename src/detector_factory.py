# src/detector_factory.py
# Sélecteur de détecteur : retourne le bon wrapper selon le type demandé.
# Permet de basculer entre YOLOv3 (OpenCV DNN) et YOLOv8 (ultralytics)
# sans changer le pipeline.
#
# Usage :
#   from src.detector_factory import build_detector
#   detector = build_detector("yolov3")   # ou "yolov8"
#   detections = detector.detect(frame)   # interface identique

import os
from pathlib import Path


# Chemins de config par défaut pour chaque détecteur
DEFAULT_CONFIGS = {
    "yolov3": "configs/yolov3.yaml",
    "yolov8": "configs/yolov8.yaml",
}


def build_detector(detector_type: str, config_path: str = None):
    """
    Construit et retourne un détecteur du type demandé.

    Paramètres
    ----------
    detector_type : str
        "yolov3" ou "yolov8"
    config_path : str, optionnel
        Chemin vers un fichier de config. Si None, utilise le défaut
        correspondant au type de détecteur.

    Retourne
    --------
    Un objet détecteur exposant la méthode .detect(frame_bgr) qui
    renvoie un np.ndarray (N, 5) : [x1, y1, x2, y2, confidence].
    """
    detector_type = detector_type.lower()

    if config_path is None:
        config_path = DEFAULT_CONFIGS.get(detector_type)
        if config_path is None:
            raise ValueError(
                f"Type de détecteur inconnu : '{detector_type}'. "
                f"Options : {list(DEFAULT_CONFIGS.keys())}"
            )

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config introuvable : {config_path}")

    if detector_type == "yolov3":
        from src.detector import YOLOv3Detector
        return YOLOv3Detector(config_path)

    elif detector_type == "yolov8":
        from src.detector_yolov8 import YOLOv8Detector
        return YOLOv8Detector(config_path)

    else:
        raise ValueError(
            f"Type de détecteur inconnu : '{detector_type}'. "
            f"Options : {list(DEFAULT_CONFIGS.keys())}"
        )
