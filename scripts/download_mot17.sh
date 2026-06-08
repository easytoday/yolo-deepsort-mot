#!/usr/bin/env bash
# scripts/download_mot17.sh
# Aide à l'installation du dataset MOT17
#
# motchallenge.net n'étant plus disponible, ce script gère :
#   - L'extraction d'une archive téléchargée depuis une source alternative
#   - La vérification de la structure après extraction
#
# Usage :
#   # Fournir le zip téléchargé manuellement :
#   bash scripts/download_mot17.sh --file /chemin/vers/MOT17.zip
#
#   # Vérifier une extraction existante :
#   bash scripts/download_mot17.sh --check
#
#   # Afficher les sources alternatives :
#   bash scripts/download_mot17.sh --sources

set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data/MOT17"
FILE_ARG=""

# Parsing des arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --file)    FILE_ARG="$2"; shift 2 ;;
        --check)   MODE="check";  shift   ;;
        --sources) MODE="sources"; shift  ;;
        *) echo "Argument inconnu : $1"; exit 1 ;;
    esac
done

echo "================================================="
echo "  MOT17 Dataset — Installation"
echo "================================================="
echo ""

# ------------------------------------------------------------------
# Sources alternatives (motchallenge.net indisponible)
# ------------------------------------------------------------------
show_sources() {
    echo "  motchallenge.net étant indisponible, voici une des sources alternatives :"
    echo ""
    echo "  Rwth Aachen University : https://www.vision.rwth-aachen.de/page/mots"
    echo "     https://www.vision.rwth-aachen.de/media/resource_files/MOTSChallenge.zip"
    echo "     → Télécharger MMOTSChallenge.zip dans la section MOTSChallenge"
    echo "     → toujours accessible en juin 2026"
    echo ""
    echo "  Une fois le fichier téléchargé :"
    echo "     bash scripts/download_mot17.sh --file /chemin/vers/MOT17.zip"
    echo ""
    echo "================================================="
}

if [ "${MODE:-}" = "sources" ]; then
    show_sources
    exit 0
fi

# ------------------------------------------------------------------
# Vérification de la structure (appelée en interne ou directement)
# ------------------------------------------------------------------
check_structure() {
    echo "  Vérification de la structure MOT17..."

    if [ ! -d "$DATA_DIR" ]; then
        echo "  ERREUR : data/MOT17/ absent"
        exit 1
    fi

    # MOT17 peut être extrait avec ou sans sous-dossier train/
    # Certaines archives ont la structure MOT17/train/MOT17-xx
    # D'autres ont directement train/MOT17-xx
    # On normalise vers data/MOT17/train/

    # Cas : extraction a créé data/MOT17/MOT17/train/
    if [ -d "$DATA_DIR/MOT17/train" ]; then
        echo "  Réorganisation : déplacement de MOT17/MOT17/ → MOT17/"
        mv "$DATA_DIR/MOT17/train" "$DATA_DIR/train" 2>/dev/null || true
        mv "$DATA_DIR/MOT17/test"  "$DATA_DIR/test"  2>/dev/null || true
        rm -rf "$DATA_DIR/MOT17"
    fi

    # Cas : extraction a créé data/MOT17/MOT17Det/ (variante du benchmark)
    if [ ! -d "$DATA_DIR/train" ] && [ -d "$DATA_DIR/MOT17Det" ]; then
        mv "$DATA_DIR/MOT17Det" "$DATA_DIR/train"
    fi

    if [ ! -d "$DATA_DIR/train" ]; then
        echo ""
        echo "  ERREUR : data/MOT17/train/ toujours absent après extraction."
        echo "  Vérifiez la structure de votre archive avec :"
        echo "    unzip -l votre_archive.zip | head -30"
        echo ""
        echo "  Structure attendue dans l'archive :"
        echo "    MOT17/train/MOT17-02-DPM/"
        echo "    MOT17/train/MOT17-02-FRCNN/"
        echo "    ..."
        exit 1
    fi

    N_TRAIN=$(ls -d "$DATA_DIR/train"/MOT17-* 2>/dev/null | wc -l)
    N_TEST=0
    [ -d "$DATA_DIR/test" ] && \
        N_TEST=$(ls -d "$DATA_DIR/test"/MOT17-* 2>/dev/null | wc -l)

    echo ""
    echo "  ✓ Structure valide :"
    echo "    Train : $N_TRAIN séquence(s)"
    echo "    Test  : $N_TEST séquence(s)"

    # Afficher les séquences trouvées
    if [ "$N_TRAIN" -gt 0 ]; then
        echo ""
        echo "  Séquences train détectées :"
        ls -d "$DATA_DIR/train"/MOT17-* | xargs -I{} basename {} | sed 's/^/    /'
    fi

    if [ "$N_TRAIN" -eq 0 ]; then
        echo "  ERREUR : Aucune séquence MOT17-* dans train/"
        exit 1
    fi

    echo ""
    echo "  ✓ MOT17 prêt à l'emploi."
    echo "  Lancer le pipeline :"
    echo "    python run_tracker.py --sequence data/MOT17/train/MOT17-02-DPM"
    echo "================================================="
}

if [ "${MODE:-}" = "check" ]; then
    check_structure
    exit 0
fi

# ------------------------------------------------------------------
# Extraction depuis un fichier fourni manuellement
# ------------------------------------------------------------------
if [ -n "$FILE_ARG" ]; then
    if [ ! -f "$FILE_ARG" ]; then
        echo "  ERREUR : Fichier introuvable : $FILE_ARG"
        exit 1
    fi

    EXT="${FILE_ARG##*.}"
    mkdir -p "$DATA_DIR"

    echo "[1/2] Extraction de $(basename "$FILE_ARG")..."

    case "$EXT" in
        zip)
            unzip -q "$FILE_ARG" -d "$DATA_DIR"
            ;;
        gz|tgz)
            tar -xzf "$FILE_ARG" -C "$DATA_DIR"
            ;;
        bz2)
            tar -xjf "$FILE_ARG" -C "$DATA_DIR"
            ;;
        *)
            echo "  ERREUR : format non supporté (attendu : .zip, .tar.gz, .tar.bz2)"
            exit 1
            ;;
    esac

    echo "  ✓ Extraction terminée dans $DATA_DIR"
    echo ""
    echo "[2/2] Vérification de la structure..."
    check_structure
    exit 0
fi

# ------------------------------------------------------------------
# Aide par défaut (aucun argument)
# ------------------------------------------------------------------
echo "  Usage :"
echo ""
echo "  1. Obtenir le dataset (motchallenge.net indisponible) :"
echo "     bash scripts/download_mot17.sh --sources"
echo ""
echo "  2. Extraire l'archive téléchargée :"
echo "     bash scripts/download_mot17.sh --file /chemin/vers/MOT17.zip"
echo ""
echo "  3. Vérifier une extraction existante :"
echo "     bash scripts/download_mot17.sh --check"
echo ""
echo "================================================="
