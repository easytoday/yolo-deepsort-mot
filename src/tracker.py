# src/tracker.py
# Wrapper Deep SORT pour le suivi multi-objets inter-frames
# Utilise l'implémentation originale de nwojke/deep_sort
# Interface : reçoit les détections YOLOv3, retourne les pistes avec IDs

import sys
import os
import numpy as np
import cv2
import yaml


class DeepSORTTracker:
    """
    Wrapper autour de Deep SORT (nwojke/deep_sort).
    Rôle : associer les détections YOLOv3 successives pour maintenir
    des identités persistantes (tracks) à travers les frames.

    Format d'entrée  : détections [x1, y1, x2, y2, conf] (coordonnées absolues)
    Format de sortie : pistes     [x1, y1, x2, y2, track_id]
    """

    def __init__(self, config_path: str):
        """
        Paramètres
        ----------
        config_path : str
            Chemin vers configs/deepsort.yaml
        """
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        reid_model_path     = self.cfg["model"]["reid_model"]
        max_cosine_distance = self.cfg["tracker"]["max_cosine_distance"]
        nn_budget           = self.cfg["tracker"]["nn_budget"]
        max_age             = self.cfg["tracker"]["max_age"]
        n_init              = self.cfg["tracker"]["n_init"]

        if not os.path.exists(reid_model_path):
            raise FileNotFoundError(
                f"Modèle ReID introuvable : {reid_model_path}\n"
                "Lancez d'abord : bash scripts/setup.sh"
            )

        self._load_tracker(
            reid_model_path,
            max_cosine_distance,
            nn_budget,
            max_age,
            n_init,
        )

    # ------------------------------------------------------------------
    # Chargement du tracker
    # ------------------------------------------------------------------

    def _load_tracker(
        self,
        reid_model_path: str,
        max_cosine_distance: float,
        nn_budget: int,
        max_age: int,
        n_init: int,
    ):
        """
        Initialise Deep SORT avec le modèle ReID et les hyperparamètres.
        Le dossier deep_sort/ doit être cloné et dans sys.path.
        """
        try:
            # Imports depuis nwojke/deep_sort
            from deep_sort import nn_matching                  # noqa
            from deep_sort.detection import Detection          # noqa
            from deep_sort.tracker import Tracker              # noqa
            from tools import generate_detections as gdet     # noqa

            self._Detection = Detection
            self._Tracker   = Tracker

            # Extracteur de features d'apparence (réseau ReID)
            self._encoder = gdet.create_box_encoder(
                reid_model_path, batch_size=1
            )

            # Métrique cosinus pour l'association par apparence
            metric = nn_matching.NearestNeighborDistanceMetric(
                "cosine", max_cosine_distance, nn_budget
            )

            # Instanciation du tracker Deep SORT
            self.tracker = Tracker(metric, max_age=max_age, n_init=n_init)
            print(f"[Tracker] Deep SORT chargé (max_age={max_age}, n_init={n_init})")

        except ImportError as e:
            raise ImportError(
                "Impossible d'importer Deep SORT.\n"
                "Vérifiez que deep_sort/ est cloné et dans sys.path.\n"
                f"Détail : {e}"
            )

    # ------------------------------------------------------------------
    # Conversion du format
    # ------------------------------------------------------------------

    @staticmethod
    def _xyxy_to_xywh(boxes_xyxy: np.ndarray) -> np.ndarray:
        """
        Convertit des boîtes [x1, y1, x2, y2] → [x, y, w, h]
        où (x, y) est le coin supérieur gauche.
        Deep SORT attend ce format en entrée.
        """
        boxes_xywh = np.empty_like(boxes_xyxy)
        boxes_xywh[:, 0] = boxes_xyxy[:, 0]                        # x1
        boxes_xywh[:, 1] = boxes_xyxy[:, 1]                        # y1
        boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]    # w
        boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]    # h
        return boxes_xywh

    # ------------------------------------------------------------------
    # Mise à jour principale
    # ------------------------------------------------------------------

    def update(
        self,
        detections_xyxy: np.ndarray,
        frame_bgr: np.ndarray,
    ) -> np.ndarray:
        """
        Met à jour le tracker avec les nouvelles détections d'une frame.

        Paramètres
        ----------
        detections_xyxy : np.ndarray, shape (N, 5)
            Détections YOLOv3 : [x1, y1, x2, y2, confidence]
        frame_bgr : np.ndarray
            Frame courante (nécessaire pour extraire les features ReID)

        Retourne
        --------
        tracks : np.ndarray, shape (M, 5)
            Pistes confirmées : [x1, y1, x2, y2, track_id]
            Uniquement les pistes "confirmed" (ayant passé n_init frames).
        """
        if detections_xyxy.shape[0] == 0:
            # Aucune détection : on fait quand même avancer le tracker
            # pour incrémenter max_age et supprimer les pistes perdues
            self.tracker.predict()
            self.tracker.update([])
            return np.empty((0, 5), dtype=np.float32)

        boxes_xywh = self._xyxy_to_xywh(detections_xyxy[:, :4])
        confidences = detections_xyxy[:, 4]

        # Extraction des features d'apparence via le réseau ReID
        # L'encodeur attend des boîtes [x, y, w, h] et retourne
        # un vecteur de 128 dimensions par boîte
        features = self._encoder(frame_bgr, boxes_xywh)

        # Construction des objets Detection pour Deep SORT
        dets = [
            self._Detection(bbox, conf, feat)
            for bbox, conf, feat in zip(boxes_xywh, confidences, features)
        ]

        # Étape 1 : prédiction via le Filtre de Kalman
        self.tracker.predict()

        # Étape 2 : association détections ↔ pistes (algorithme hongrois)
        self.tracker.update(dets)

        # Collecte des pistes confirmées
        tracks_out = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # Récupérer la boîte en format [x1, y1, x2, y2]
            bbox = track.to_tlbr()   # top-left bottom-right
            track_id = track.track_id

            tracks_out.append([
                bbox[0], bbox[1], bbox[2], bbox[3], track_id
            ])

        if len(tracks_out) == 0:
            return np.empty((0, 5), dtype=np.float32)

        return np.array(tracks_out, dtype=np.float32)

    def reset(self):
        """
        Réinitialise le tracker entre deux séquences.
        Important : les IDs doivent repartir de 1 à chaque séquence MOT17.
        """
        self.tracker.tracks.clear()
        self.tracker._next_id = 1
        print("[Tracker] Réinitialisé")
