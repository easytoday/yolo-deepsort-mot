# generate_figures.py
# Génère les graphiques de résultats pour le rapport LaTeX
# à partir des fichiers CSV produits par evaluate.py et run_ablation.py
#
# Produit (dans figures/ pour le français, figures_en/ pour l'anglais) :
#   - mota_par_sequence.pdf    : MOTA par séquence (baseline)
#   - ablation_mota_idf1.pdf   : comparatif MOTA + IDF1 des 8 configurations
#   - fp_fn_comparison.pdf     : compromis faux positifs / faux négatifs
#   - comparaison_detecteurs.pdf : YOLOv3 vs YOLOv8 (meilleures configs)
#
# Usage :
#   python generate_figures.py            # les deux langues (fr + en)
#   python generate_figures.py --lang fr  # français seulement -> figures/
#   python generate_figures.py --lang en  # anglais seulement  -> figures_en/
#
# Les fichiers PDF sont vectoriels (ok pour rapport LaTeX).
# Pour du PNG, changer l'extension dans les appels savefig().

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).parent

# ------------------------------------------------------------------
# Style sobre et académique
# ------------------------------------------------------------------
mpl.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

# Palette sobre
COL_MOTA = "#2c6fbb"   # bleu
COL_IDF1 = "#d98c34"   # orange
COL_BASE = "#9aa0a6"   # gris (baseline)

# ------------------------------------------------------------------
# Chaînes de caractères localisées (fr / en)
# ------------------------------------------------------------------
# Seul le texte affiché change d'une langue à l'autre ; les données,
# les couleurs et la mise en forme restent strictement identiques.
STRINGS = {
    "fr": {
        "out_dir": ROOT / "figures",
        "pretty": {
            "baseline": "référence", "res_608": "rés. 608", "res_832": "rés. 832",
            "conf_04": "conf. 0,4", "conf_03": "conf. 0,3",
            "ds_age50": "age 50", "ds_age70": "age 70", "combo_best": "combinée",
        },
        "seq_xlabel": "Séquence",
        "mota_title": "MOTA par séquence — configuration de référence",
        "config_xlabel": "Configuration",
        "score_ylabel": "Score (%)",
        "ablation_title": "Comparaison des configurations — MOTA et IDF1",
        "errors_ylabel": "Nombre d'erreurs",
        "fp_label": "Faux positifs (FP)",
        "fn_label": "Faux négatifs (FN)",
        "fpfn_title": "Compromis faux positifs / faux négatifs",
        "det_labels": ["YOLOv3\n(combinée)", "YOLOv8\n(combinée)"],
        "det_title": "YOLOv3 vs YOLOv8 — meilleures configurations",
    },
    "en": {
        "out_dir": ROOT / "figures_en",
        "pretty": {
            "baseline": "baseline", "res_608": "res. 608", "res_832": "res. 832",
            "conf_04": "conf. 0.4", "conf_03": "conf. 0.3",
            "ds_age50": "age 50", "ds_age70": "age 70", "combo_best": "combined",
        },
        "seq_xlabel": "Sequence",
        "mota_title": "MOTA per sequence — baseline configuration",
        "config_xlabel": "Configuration",
        "score_ylabel": "Score (%)",
        "ablation_title": "Configuration comparison — MOTA and IDF1",
        "errors_ylabel": "Number of errors",
        "fp_label": "False positives (FP)",
        "fn_label": "False negatives (FN)",
        "fpfn_title": "False positive / false negative trade-off",
        "det_labels": ["YOLOv3\n(combined)", "YOLOv8\n(combined)"],
        "det_title": "YOLOv3 vs YOLOv8 — best configurations",
    },
}


# ------------------------------------------------------------------
# Figure 1 : MOTA par séquence (baseline)
# ------------------------------------------------------------------

def figure_mota_par_sequence(S):
    """
    Diagramme en barres du MOTA par séquence pour la configuration de référence
    (YOLOv3 baseline). Lit results/ablation/baseline/metrics.csv, source fiable
    des valeurs YOLOv3 par séquence (results/mot17/metrics_train.csv peut avoir
    été écrasé par des tests ultérieurs).
    """
    csv_path = ROOT / "results" / "ablation" / "baseline" / "metrics.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0)
        # Exclure la ligne OVERALL pour le graphe par séquence
        seq_df = df[df.index != "OVERALL"]
        sequences = [s.replace("-DPM", "") for s in seq_df.index]
        mota = (seq_df["mota"] * 100).tolist()
    else:
        # Valeurs de secours (YOLOv3 baseline connues) si le CSV est absent
        print(f"  [INFO] {csv_path} absent, utilisation des valeurs connues")
        sequences = ["MOT17-02", "MOT17-05", "MOT17-09", "MOT17-11"]
        mota = [39.9, 61.3, 63.2, 64.3]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(sequences, mota, color=COL_MOTA, width=0.6, edgecolor="black", linewidth=0.5)

    # Étiquettes de valeur au-dessus des barres
    for bar, val in zip(bars, mota):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("MOTA (%)")
    ax.set_xlabel(S["seq_xlabel"])
    ax.set_ylim(0, max(mota) * 1.15)
    ax.set_title(S["mota_title"], fontsize=12)

    plt.tight_layout()
    out = S["out_dir"] / "mota_par_sequence.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ------------------------------------------------------------------
# Figure 2 : Comparatif MOTA + IDF1 des configurations (ablation)
# ------------------------------------------------------------------

def figure_ablation(S):
    """
    Diagramme en barres groupées MOTA + IDF1 pour les 8 configurations.
    Lit results/ablation/comparison.csv.
    """
    csv_path = ROOT / "results" / "ablation" / "comparison.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        labels = df["exp"].tolist()
        mota = (df["MOTA"] * 100).tolist()
        idf1 = (df["IDF1"] * 100).tolist()
    else:
        print(f"  [INFO] {csv_path} absent, utilisation des valeurs connues")
        labels = ["baseline", "res_608", "res_832", "conf_04",
                  "conf_03", "ds_age50", "ds_age70", "combo_best"]
        mota = [57.0, 60.6, 59.0, 57.1, 56.6, 56.8, 56.8, 60.3]
        idf1 = [54.4, 57.4, 58.5, 54.8, 54.0, 58.6, 58.7, 62.0]

    # Noms plus lisibles pour l'affichage
    labels = [S["pretty"].get(l, l) for l in labels]

    x = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars1 = ax.bar([i - width/2 for i in x], mota, width,
                   label="MOTA", color=COL_MOTA, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar([i + width/2 for i in x], idf1, width,
                   label="IDF1", color=COL_IDF1, edgecolor="black", linewidth=0.5)

    # Étiquettes de valeur
    for bars, vals in [(bars1, mota), (bars2, idf1)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel(S["score_ylabel"])
    ax.set_xlabel(S["config_xlabel"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(50, 66)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title(S["ablation_title"], fontsize=12)

    # Ligne de référence (baseline MOTA)
    ax.axhline(y=mota[0], color=COL_BASE, linestyle=":", linewidth=1, alpha=0.7)

    plt.tight_layout()
    out = S["out_dir"] / "ablation_mota_idf1.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ------------------------------------------------------------------
# Figure 3 : FP vs FN par configuration
# ------------------------------------------------------------------

def figure_fp_fn(S):
    """
    Visualise le compromis FP/FN selon la configuration.
    Illustre l'effet pervers de res_832 (FN bas mais FP haut).
    """
    csv_path = ROOT / "results" / "ablation" / "comparison.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        labels = df["exp"].tolist()
        fp = df["FP"].tolist()
        fn = df["FN"].tolist()
    else:
        print(f"  [INFO] {csv_path} absent, utilisation des valeurs connues")
        labels = ["baseline", "res_608", "res_832", "conf_04",
                  "conf_03", "ds_age50", "ds_age70", "combo_best"]
        fp = [1311, 1841, 2883, 1802, 2417, 1274, 1307, 2362]
        fn = [10077, 8564, 7941, 9539, 9038, 10176, 10141, 8122]

    labels = [S["pretty"].get(l, l) for l in labels]

    x = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - width/2 for i in x], fp, width,
           label=S["fp_label"], color="#c0504d", edgecolor="black", linewidth=0.5)
    ax.bar([i + width/2 for i in x], fn, width,
           label=S["fn_label"], color="#4f81bd", edgecolor="black", linewidth=0.5)

    ax.set_ylabel(S["errors_ylabel"])
    ax.set_xlabel(S["config_xlabel"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(S["fpfn_title"], fontsize=12)

    plt.tight_layout()
    out = S["out_dir"] / "fp_fn_comparison.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ------------------------------------------------------------------
# Figure 4 : Comparaison YOLOv3 vs YOLOv8 (meilleures configs)
# ------------------------------------------------------------------

def figure_comparaison_detecteurs(S):
    """
    Compare les meilleures configurations YOLOv3 et YOLOv8 sur MOTA et IDF1.
    Lit results/ablation/comparison.csv.
    """
    csv_path = ROOT / "results" / "ablation" / "comparison.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Meilleure config de chaque détecteur (par MOTA)
        best = {}
        for det in ["yolov3", "yolov8"]:
            sub = df[df["detecteur"] == det]
            if len(sub) > 0:
                best[det] = sub.loc[sub["MOTA"].idxmax()]
    else:
        print(f"  [INFO] {csv_path} absent, utilisation des valeurs connues")
        best = None

    labels = S["det_labels"]
    if best and "yolov3" in best and "yolov8" in best:
        mota = [best["yolov3"]["MOTA"] * 100, best["yolov8"]["MOTA"] * 100]
        idf1 = [best["yolov3"]["IDF1"] * 100, best["yolov8"]["IDF1"] * 100]
    else:
        mota = [60.3, 62.6]
        idf1 = [62.0, 65.0]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars1 = ax.bar([i - width/2 for i in x], mota, width,
                   label="MOTA", color=COL_MOTA, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar([i + width/2 for i in x], idf1, width,
                   label="IDF1", color=COL_IDF1, edgecolor="black", linewidth=0.5)

    for bars, vals in [(bars1, mota), (bars2, idf1)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel(S["score_ylabel"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(55, 68)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title(S["det_title"], fontsize=12)

    plt.tight_layout()
    out = S["out_dir"] / "comparaison_detecteurs.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ------------------------------------------------------------------
# Programme principal
# ------------------------------------------------------------------

def generate_for_language(lang):
    S = STRINGS[lang]
    S["out_dir"].mkdir(exist_ok=True)
    print(f"\n[Figures] Langue « {lang} » — dossier de sortie : {S['out_dir']}/\n")
    figure_mota_par_sequence(S)
    figure_ablation(S)
    figure_fp_fn(S)
    figure_comparaison_detecteurs(S)


def main():
    parser = argparse.ArgumentParser(
        description="Génère les graphiques du rapport (français et/ou anglais).")
    parser.add_argument("--lang", choices=["fr", "en", "both"], default="both",
                        help="Langue des figures (défaut : both).")
    args = parser.parse_args()

    langs = ["fr", "en"] if args.lang == "both" else [args.lang]

    print("\n[Figures] Génération des graphiques du rapport...")
    for lang in langs:
        generate_for_language(lang)

    print(f"\n  Terminé. Insérer dans le rapport LaTeX avec :")
    print(f"  \\includegraphics[width=0.8\\textwidth]{{figures/mota_par_sequence.pdf}}     % FR")
    print(f"  \\includegraphics[width=0.8\\textwidth]{{figures_en/mota_par_sequence.pdf}}  % EN")
    print()


if __name__ == "__main__":
    main()
