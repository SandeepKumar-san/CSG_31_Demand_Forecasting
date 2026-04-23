"""
Generate Fig. 2: Combined 2-panel alpha trajectory for IEEE paper.
Left: SupplyGraph (α converges fast to ~0.92, tight band)
Right: USGS (α oscillates ~0.5, wide band)
Saves to: results/fig2_alpha_trajectory.pdf and .png
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

SEEDS = [42, 123, 456, 789, 1337]
SG_LOG_DIR  = r"s:\Demand_Forecast\results\supplygraph\logs"
USGS_LOG_DIR = r"s:\Demand_Forecast\results\usgs\logs"
OUT_DIR = r"s:\Demand_Forecast\results"

def load_alpha(log_dir, seeds):
    arrays = []
    for s in seeds:
        p = os.path.join(log_dir, f"training_history_seed{s}.json")
        with open(p) as f:
            d = json.load(f)
        arrays.append(np.array(d["alpha_mean"]))
    # Align to shortest run
    min_len = min(len(a) for a in arrays)
    arrays = [a[:min_len] for a in arrays]
    return np.array(arrays)  # shape (5, epochs)

sg   = load_alpha(SG_LOG_DIR,   SEEDS)
usgs = load_alpha(USGS_LOG_DIR, SEEDS)

epochs_sg   = np.arange(1, sg.shape[1]+1)
epochs_usgs = np.arange(1, usgs.shape[1]+1)

mean_sg, std_sg     = sg.mean(0),   sg.std(0)
mean_usgs, std_usgs = usgs.mean(0), usgs.std(0)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=False)
plt.rcParams['font.family'] = 'DejaVu Sans'

BLUE  = '#1565C0'
FILL  = '#90CAF9'
GREY  = '#757575'

for ax, mean, std, epochs, title, final_ann, ylim in [
    (axes[0], mean_sg,   std_sg,   epochs_sg,
     '(a) SupplyGraph',
     f'$\\bar{{\\alpha}}={mean_sg[-1]:.3f}$',
     (0.0, 1.05)),
    (axes[1], mean_usgs, std_usgs, epochs_usgs,
     '(b) USGS Mineral Commodity',
     f'$\\bar{{\\alpha}}={mean_usgs[-1]:.3f}$',
     (0.0, 1.05)),
]:
    ax.fill_between(epochs, mean-std, mean+std, color=FILL, alpha=0.55, label='$\\pm$1 Std (5 seeds)')
    ax.plot(epochs, mean, color=BLUE, lw=1.8, label='Mean $\\alpha$')
    ax.axhline(0.5, color=GREY, lw=1.0, ls='--', label='Equal fusion ($\\alpha=0.5$)')
    # Annotate final value
    ax.annotate(final_ann, xy=(epochs[-1], mean[-1]),
                xytext=(-38, 8), textcoords='offset points',
                fontsize=7.5, color=BLUE,
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.8))
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(*ylim)
    ax.set_xlabel('Epoch', fontsize=8)
    ax.set_ylabel('Fusion weight $\\alpha$', fontsize=8)
    ax.set_title(title, fontsize=8.5, fontweight='bold', pad=4)
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Shared legend on bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3,
           fontsize=7, frameon=False,
           bbox_to_anchor=(0.5, -0.04))

fig.suptitle('Fusion Weight ($\\alpha$) Evolution Across Five Seeds',
             fontsize=9, fontweight='bold', y=1.01)
plt.tight_layout(rect=[0, 0.07, 1, 1])

for ext in ('pdf', 'png'):
    out = os.path.join(OUT_DIR, f"fig2_alpha_trajectory.{ext}")
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved: {out}")

print("Done.")
