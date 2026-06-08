#!/usr/bin/env bash
# make_gif.sh
# Convertit une vidéo .avi annotée en GIF animé léger pour le README GitHub.
# Utilise ffmpeg avec une palette optimisée (256 couleurs) pour minimiser le poids.
#
# Réglages : extrait de ~10s, largeur 640px, 12 fps → typiquement 3-4 Mo
#
# Usage :
#   bash make_gif.sh <video.avi> <sortie.gif> [début_secondes] [durée_secondes]
#
# Exemples :
#   bash make_gif.sh videos/mot17-02_yolov8.avi docs/yolov8.gif
#   bash make_gif.sh videos/mot17-02_yolov3.avi docs/yolov3.gif 5 10

set -e

INPUT="$1"
OUTPUT="$2"
START="${3:-0}"      # début de l'extrait (secondes), défaut 0
DURATION="${4:-10}"  # durée de l'extrait (secondes), défaut 10

# Paramètres de compression
WIDTH=640            # largeur cible (hauteur auto pour garder le ratio)
FPS=12               # images par seconde du GIF

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
    echo "Usage : bash make_gif.sh <video.avi> <sortie.gif> [début_s] [durée_s]"
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo "ERREUR : fichier introuvable : $INPUT"
    exit 1
fi

# Vérifier ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "ERREUR : ffmpeg n'est pas installé."
    echo "  Debian/Ubuntu : sudo apt install ffmpeg"
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"

# Fichier temporaire pour la palette
PALETTE="$(mktemp --suffix=.png)"

echo "[GIF] Source   : $INPUT"
echo "[GIF] Extrait  : ${START}s → +${DURATION}s"
echo "[GIF] Cible    : ${WIDTH}px de large, ${FPS} fps"
echo ""

# Filtre commun : extraire, réduire fps, redimensionner
FILTERS="fps=${FPS},scale=${WIDTH}:-1:flags=lanczos"

# Passe 1 : générer une palette optimale à partir de l'extrait
echo "[GIF] Passe 1/2 : génération de la palette..."
ffmpeg -v warning -ss "$START" -t "$DURATION" -i "$INPUT" \
    -vf "${FILTERS},palettegen=stats_mode=diff" \
    -y "$PALETTE"

# Passe 2 : appliquer la palette pour produire le GIF
echo "[GIF] Passe 2/2 : encodage du GIF..."
ffmpeg -v warning -ss "$START" -t "$DURATION" -i "$INPUT" -i "$PALETTE" \
    -lavfi "${FILTERS} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
    -y "$OUTPUT"

rm -f "$PALETTE"

# Afficher le poids final
SIZE=$(du -h "$OUTPUT" | cut -f1)
echo ""
echo "[GIF]  Généré : $OUTPUT ($SIZE)"
