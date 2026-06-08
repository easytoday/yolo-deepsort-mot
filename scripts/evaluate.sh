#!/usr/bin/env bash
# scripts/evaluate.sh
# Enchaîne le tracking batch + l'évaluation des métriques
# Usage : bash scripts/evaluate.sh [train|test]
#
# Prérequis :
#   - conda activate yolo_deepsort
#   - bash scripts/setup.sh  (poids et repos clonés)
#   - data/MOT17/ présent

set -e

SPLIT="${1:-train}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "================================================="
echo "  Évaluation complète MOT17 / split=$SPLIT"
echo "================================================="

# Étape 1 : Tracking sur toutes les séquences
echo ""
echo "[1/2] Tracking (YOLOv3 + Deep SORT)..."
python run_mot17.py --split "$SPLIT"

# Étape 2 : Calcul des métriques
echo ""
echo "[2/2] Calcul des métriques (MOTA, MOTP, IDF1)..."
python evaluate.py --split "$SPLIT"

echo ""
echo "  Évaluation terminée."
echo "  Résultats : results/mot17/metrics_${SPLIT}.csv"
