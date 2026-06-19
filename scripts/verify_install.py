# verify_install.py
# Vérifie que tous les prérequis sont bien en place avant de lancer le pipeline
# Usage : python verify_install.py

import sys
import os
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deep_sort"))

# Couleurs terminal
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

OK   = f"{GREEN}  ✓{RESET}"
WARN = f"{YELLOW}  ⚠{RESET}"
FAIL = f"{RED}  ✗{RESET}"

errors   = []
warnings = []


def check(label: str, fn):
    try:
        result = fn()
        msg = f" ({result})" if result else ""
        print(f"{OK} {label}{msg}")
        return True
    except Warning as w:
        warnings.append(str(w))
        print(f"{WARN} {label} — {w}")
        return True
    except Exception as e:
        errors.append(f"{label}: {e}")
        print(f"{FAIL} {label} — {e}")
        return False


# ------------------------------------------------------------------
# 1. Packages Python
# ------------------------------------------------------------------
print(f"\n{BOLD}[1/6] Packages Python{RESET}")

def check_cv2():
    import cv2
    assert hasattr(cv2, "dnn"), "Module cv2.dnn absent"
    return f"cv2 {cv2.__version__}"

def check_numpy():
    import numpy as np
    return f"numpy {np.__version__}"

def check_filterpy():
    import filterpy
    return f"filterpy {filterpy.__version__}"

def check_lap():
    import lap
    return "lap OK"

def check_motmetrics():
    import motmetrics as mm
    return f"motmetrics {mm.__version__}"

def check_scipy():
    import scipy
    return f"scipy {scipy.__version__}"

def check_yaml():
    import yaml
    return "pyyaml OK"

for label, fn in [
    ("opencv-python (cv2 + DNN)", check_cv2),
    ("numpy",                      check_numpy),
    ("filterpy (Filtre de Kalman)", check_filterpy),
    ("lap (algorithme hongrois)",   check_lap),
    ("motmetrics",                  check_motmetrics),
    ("scipy",                       check_scipy),
    ("pyyaml",                      check_yaml),
]:
    check(label, fn)


# ------------------------------------------------------------------
# 2. Fichiers poids
# ------------------------------------------------------------------
print(f"\n{BOLD}[2/6] Fichiers poids{RESET}")

def check_file(path, size_min_bytes=0):
    """Vérifie la présence et la taille minimale d'un fichier (en octets)."""
    def _check():
        p = ROOT / path
        if not p.exists():
            raise Exception(f"Introuvable : {p}")
        size_bytes = p.stat().st_size
        if size_bytes < size_min_bytes:
            raise Exception(
                f"Fichier trop petit ({size_bytes} octets < {size_min_bytes} attendus)"
                f" — téléchargement incomplet ?"
            )
        size_mo = size_bytes / 1e6
        return f"{size_mo:.1f} Mo" if size_mo >= 0.1 else f"{size_bytes} octets"
    return _check

check("weights/yolov3.weights",   check_file("weights/yolov3.weights",   size_min_bytes=200_000_000))
check("weights/yolov3.cfg",       check_file("weights/yolov3.cfg",        size_min_bytes=5_000))
check("weights/coco.names",       check_file("weights/coco.names",        size_min_bytes=100))
check("weights/mars-small128.pb", check_file("weights/mars-small128.pb",  size_min_bytes=1_000_000))


# ------------------------------------------------------------------
# 3. Repo Deep SORT
# ------------------------------------------------------------------
print(f"\n{BOLD}[3/6] Deep SORT (nwojke/deep_sort){RESET}")

def check_deepsort_repo():
    if not (ROOT / "deep_sort" / "deep_sort").exists():
        raise Exception("Dossier deep_sort/deep_sort/ absent — clonez le repo")
    return "repo présent"

def check_deepsort_imports():
    from deep_sort import nn_matching
    from deep_sort.detection import Detection
    from deep_sort.tracker import Tracker
    return "imports OK"

def check_deepsort_encoder():
    import warnings
    # Suppression des warnings TensorFlow/CUDA (normaux sans GPU)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from tools import generate_detections as gdet
    return "encoder importable"

check("deep_sort/ cloné",          check_deepsort_repo)
check("imports deep_sort",         check_deepsort_imports)
check("tools.generate_detections", check_deepsort_encoder)


# ------------------------------------------------------------------
# 4. Test YOLOv3 (inférence sur image synthétique)
# ------------------------------------------------------------------
print(f"\n{BOLD}[4/6] Test YOLOv3 (inférence CPU){RESET}")

def check_yolo_inference():
    import cv2
    cfg_path     = str(ROOT / "weights" / "yolov3.cfg")
    weights_path = str(ROOT / "weights" / "yolov3.weights")

    if not os.path.exists(cfg_path) or not os.path.exists(weights_path):
        raise Warning("Poids absents — test ignoré")

    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    img  = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    out_indices = net.getUnconnectedOutLayers()
    if isinstance(out_indices[0], (list, np.ndarray)):
        out_indices = [i[0] for i in out_indices]
    output_layers = [layer_names[i - 1] for i in out_indices]

    t0  = time.time()
    out = net.forward(output_layers)
    dt  = time.time() - t0

    total_anchors = sum(o.shape[0] for o in out)
    return f"inference en {dt:.2f}s | {total_anchors} anchors"

check("Inférence YOLOv3 sur image 416×416", check_yolo_inference)


# ------------------------------------------------------------------
# 5. Test Deep SORT (initialisation tracker)
# ------------------------------------------------------------------
print(f"\n{BOLD}[5/6] Test Deep SORT (initialisation){RESET}")

def check_tracker_init():
    import warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    reid_path = str(ROOT / "weights" / "mars-small128.pb")
    if not os.path.exists(reid_path):
        raise Warning("mars-small128.pb absent — test ignoré")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from deep_sort import nn_matching
        from deep_sort.tracker import Tracker
        from deep_sort.detection import Detection
        from tools import generate_detections as gdet

    encoder = gdet.create_box_encoder(reid_path, batch_size=1)
    metric  = nn_matching.NearestNeighborDistanceMetric("cosine", 0.3, 100)
    tracker = Tracker(metric, max_age=30, n_init=3)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes       = np.array([[100, 100, 60, 120],
                             [300, 200, 50, 100]])
    confs       = [0.9, 0.8]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = encoder(dummy_frame, boxes)

    for _ in range(3):
        dets = [Detection(b, c, f) for b, c, f in zip(boxes, confs, features)]
        tracker.predict()
        tracker.update(dets)

    confirmed = [t for t in tracker.tracks if t.is_confirmed()]
    return f"{len(confirmed)} piste(s) confirmée(s) après 3 frames"

check("Initialisation et 3 frames fictives", check_tracker_init)


# ------------------------------------------------------------------
# 6. Données MOT17 (avertissement si absentes)
# ------------------------------------------------------------------
print(f"\n{BOLD}[6/6] Données MOT17{RESET}")

def check_mot17():
    mot17_train = ROOT / "data" / "MOT17" / "train"
    if not mot17_train.exists():
        raise Warning(
            "data/MOT17/train/ absent — normal si MOT17 pas encore téléchargé\n"
            "    → bash scripts/download_mot17.sh"
        )
    seqs = [s for s in mot17_train.iterdir()
            if s.is_dir() and s.name.startswith("MOT17-")]
    if not seqs:
        raise Warning(
            "data/MOT17/train/ vide — normal si MOT17 pas encore téléchargé\n"
            "    → bash scripts/download_mot17.sh"
        )
    return f"{len(seqs)} séquence(s) détectée(s)"

check("data/MOT17/train/", check_mot17)


# ------------------------------------------------------------------
# Bilan
# ------------------------------------------------------------------
print(f"\n{'='*55}")
hard_errors = [e for e in errors if "MOT17" not in e]

if not hard_errors:
    print(f"{GREEN}{BOLD}  ✓ Système prêt — vous pouvez lancer le pipeline !{RESET}")
    if warnings:
        print(f"\n{YELLOW}  Avertissements (non bloquants) :{RESET}")
        for w in warnings:
            for line in w.split("\n"):
                print(f"    {line}")
else:
    print(f"{RED}{BOLD}  ✗ {len(hard_errors)} erreur(s) bloquante(s) :{RESET}")
    for e in hard_errors:
        print(f"    {RED}→ {e}{RESET}")
    print(f"\n  Correction : pip install filterpy lap motmetrics pyyaml")
    if warnings:
        print(f"\n{YELLOW}  Avertissements :{RESET}")
        for w in warnings:
            print(f"    - {w}")
print(f"{'='*55}\n")

sys.exit(1 if hard_errors else 0)
