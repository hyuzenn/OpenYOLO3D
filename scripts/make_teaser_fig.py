"""Teaser figure for the OV-TCS paper. All numbers are from the frozen main
table (Tab. 1) — no new values."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ego, glob = '#8f8f8f', '#0072B2'
panels = [
    ("mAP", [0.3407, 0.3407], "bit-identical", "%.4f", (0, 0.50)),
    ("OV-TCS", [0.136, 0.188], "+38%", "%.3f", (0, 0.26)),
    ("Fragmentation (lower better)", [10.64, 4.45], "−58%", "%.2f", (0, 14.5)),
]
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.1))
for ax, (title, vals, note, fmt, ylim) in zip(axes, panels):
    bars = ax.bar([0, 1], vals, width=0.62, color=[ego, glob], zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt % v, ha='center',
                va='bottom', fontsize=8, color='#333333')
    ax.set_title(title, fontsize=9.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Ego', 'Global'], fontsize=9)
    ax.set_ylim(*ylim)
    ax.tick_params(axis='y', labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#e6e6e6', lw=0.6)
    ax.text(0.5, 0.95, note, transform=ax.transAxes, ha='center', va='top',
            fontsize=8.5, style='italic', color='#555555')
fig.suptitle('Same detections, same mAP — different temporal quality',
             fontsize=10, y=1.04)
fig.tight_layout()
fig.savefig('paper/figs/fig_teaser.pdf', bbox_inches='tight')
fig.savefig('paper/figs/fig_teaser.png', dpi=200, bbox_inches='tight')
print('done')
