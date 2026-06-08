#!/usr/bin/env bash
# scripts/setup.sh
# Installation des dépendances externes :
#   - Clone nwojke/deep_sort  (tracker)
#   - Télécharge les poids YOLOv3 COCO pré-entraînés (darknet format)
#   - Télécharge le modèle ReID Deep SORT (mars-small128.pb)
#
# Usage : bash scripts/setup.sh
# Prérequis : conda activate yolo_deepsort

set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "================================================="
echo "  Setup YOLOv3 + Deep SORT — CPU / Debian 13"
echo "================================================="
echo "  Répertoire racine : $ROOT_DIR"
echo ""

# ------------------------------------------------------------------
# 1. Clone nwojke/deep_sort
# ------------------------------------------------------------------
echo "[1/3] Clonage de nwojke/deep_sort..."

if [ -d "deep_sort/.git" ]; then
    echo "  → Déjà cloné, mise à jour..."
    cd deep_sort && git pull --quiet && cd ..
else
    git clone --depth 1 https://github.com/nwojke/deep_sort.git
    echo "  nwojke/deep_sort cloné"
fi

# ------------------------------------------------------------------
# 2. Téléchargement des fichiers YOLOv3 (poids darknet + config)
# ------------------------------------------------------------------
echo ""
echo "[2/3] Téléchargement des fichiers YOLOv3..."
mkdir -p weights

YOLO_WEIGHTS_URL="https://pjreddie.com/media/files/yolov3.weights"
YOLO_CFG_URL="https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
COCO_NAMES_URL="https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

if [ ! -f "weights/yolov3.weights" ]; then
    echo "  Téléchargement yolov3.weights (~236 Mo)..."
    wget -q --show-progress -O weights/yolov3.weights "$YOLO_WEIGHTS_URL"
    echo "  yolov3.weights téléchargé"
else
    echo "  → weights/yolov3.weights déjà présent ($(du -h weights/yolov3.weights | cut -f1))"
fi

if [ ! -f "weights/yolov3.cfg" ]; then
    echo "  Téléchargement yolov3.cfg..."
    wget -q -O weights/yolov3.cfg "$YOLO_CFG_URL"
    echo "  yolov3.cfg téléchargé"
else
    echo "  → weights/yolov3.cfg déjà présent"
fi

if [ ! -f "weights/coco.names" ]; then
    echo "  Téléchargement coco.names..."
    wget -q -O weights/coco.names "$COCO_NAMES_URL"
    echo "  coco.names téléchargé"
else
    echo "  → weights/coco.names déjà présent"
fi

# ------------------------------------------------------------------
# 3. Téléchargement du modèle ReID Deep SORT (mars-small128.pb)
# ------------------------------------------------------------------
echo ""
echo "[3/3] Téléchargement du modèle ReID Deep SORT (mars-small128.pb)..."

REID_FILE="weights/mars-small128.pb"
REID_SIZE_MIN=3000000   # ~3 Mo minimum attendu

download_reid_success=0

if [ ! -f "$REID_FILE" ] || [ "$(stat -c%s "$REID_FILE" 2>/dev/null || echo 0)" -lt "$REID_SIZE_MIN" ]; then

    # --- Tentative 1 : gdown (syntaxe actuelle sans --id) ---
    echo "  Tentative 1 : gdown (Google Drive)..."
    GDRIVE_URL="https://drive.google.com/uc?id=1m2ebLHB2JThZC8vWGDYEKGsevLssSkjo"

    # Utiliser le gdown de l'environnement conda du projet (pas celui du système)
    GDOWN_BIN="$(which gdown 2>/dev/null || echo '')"
    if [ -n "$GDOWN_BIN" ]; then
        # Supprimer l'éventuel fichier partiel
        rm -f "$REID_FILE"
        if python -m gdown "$GDRIVE_URL" -O "$REID_FILE" 2>/dev/null; then
            if [ -f "$REID_FILE" ] && [ "$(stat -c%s "$REID_FILE")" -gt "$REID_SIZE_MIN" ]; then
                echo "  mars-small128.pb téléchargé via gdown"
                download_reid_success=1
            fi
        fi
    else
        echo "  gdown absent dans l'environnement conda (pip install gdown)"
    fi

    # --- Tentative 2 : wget sur mirror GitHub (ZQPei/deep_sort_pytorch) ---
    if [ "$download_reid_success" -eq 0 ]; then
        echo "  Tentative 2 : mirror GitHub (ZQPei/deep_sort_pytorch)..."
        MIRROR_URL="https://github.com/ZQPei/deep_sort_pytorch/releases/download/v1.0/ckpt.t7"
        # Note : ce mirror fournit un modèle ReID compatible (.t7 PyTorch)
        # mais notre implémentation attend le .pb TensorFlow de nwojke
        # → on ne l'utilise pas, on passe à la tentative suivante
        echo "  → Format incompatible (.t7 vs .pb), tentative ignorée"
    fi

    # --- Tentative 3 : wget sur mirror Hugging Face ---
    if [ "$download_reid_success" -eq 0 ]; then
        echo "  Tentative 3 : Hugging Face Hub..."
        HF_URL="https://huggingface.co/datasets/bpanatte/mars-small128/resolve/main/mars-small128.pb"
        rm -f "$REID_FILE"
        if wget -q --show-progress -O "$REID_FILE" "$HF_URL" 2>/dev/null; then
            if [ -f "$REID_FILE" ] && [ "$(stat -c%s "$REID_FILE")" -gt "$REID_SIZE_MIN" ]; then
                echo "  mars-small128.pb téléchargé via Hugging Face"
                download_reid_success=1
            else
                rm -f "$REID_FILE"
            fi
        fi
    fi

    # --- Tentative 4 : téléchargement manuel ---
    if [ "$download_reid_success" -eq 0 ]; then
        echo ""
        echo "  Téléchargement automatique impossible."
        echo ""
        echo "  ──────────────────────────────────────────────────"
        echo "  TÉLÉCHARGEMENT MANUEL requis pour mars-small128.pb"
        echo "  ──────────────────────────────────────────────────"
        echo ""
        echo "  Option 1 — Google Drive (navigateur) :"
        echo "    1. Ouvrez dans un navigateur :"
        echo "       https://drive.google.com/file/d/1m2ebLHB2JThZC8vWGDYEKGsevLssSkjo"
        echo "    2. Cliquez sur 'Télécharger'"
        echo "    3. Copiez le fichier dans : $ROOT_DIR/weights/mars-small128.pb"
        echo ""
        echo "  Option 2 — Depuis le repo deep_sort (si disponible) :"
        echo "    Le fichier est parfois inclus dans certains forks :"
        echo "    https://github.com/nwojke/deep_sort (dossier resources/)"
        echo ""
        echo "  Une fois le fichier placé dans weights/, relancez :"
        echo "    bash scripts/setup.sh"
        echo ""
        echo "  Setup incomplet — en attente de mars-small128.pb"
        echo "================================================="
        exit 1
    fi

else
    echo "  → weights/mars-small128.pb déjà présent ($(du -h "$REID_FILE" | cut -f1))"
fi

# ------------------------------------------------------------------
# Résumé final
# ------------------------------------------------------------------
echo ""
echo "================================================="
echo "  Setup terminé !"
echo "================================================="
echo ""
echo "  Fichiers dans weights/ :"
ls -lh weights/
echo ""
echo "  Étapes suivantes :"
echo "    python verify_install.py"
echo "    python test_pipeline.py"
echo "    bash scripts/download_mot17.sh"
echo "================================================="
