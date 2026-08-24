import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")

def box(x, y, w, h, text, color="#dbe9f6"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                        facecolor=color, edgecolor="black", linewidth=1)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9, wrap=True)

def arrow(x1, y1, x2, y2, label=""):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                         color="black", linewidth=1)
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.15, label, fontsize=7, ha="center", style="italic")

box(3.5, 5.7, 3, 0.9, "Serving LEO satellite\n(orbital propagation, TLE)")
box(3.5, 0.2, 3, 0.9, "UE @ Aberdeen ground station\n(fixed location)")
box(0.3, 3.0, 2.6, 1.0, "Link budget model\n(CNR from geometry)")
box(3.5, 3.0, 3, 1.0, "Remaining usable\nlink time estimate")
box(7.1, 3.0, 2.6, 1.0, "Uncertainty model\n(sigma_T)")
box(3.2, 1.4, 3.6, 1.0, "Conditional handover\ntrigger policy (proposed /\nbaselines)", color="#fce8d5")

arrow(5, 5.7, 5, 4.0)
arrow(5, 0.9, 5, 3.0)
arrow(2.9, 3.5, 3.5, 3.5)
arrow(6.5, 3.5, 7.1, 3.5)
arrow(5, 3.0, 5, 2.4)
arrow(7.1, 3.3, 6.8, 2.4, label="feeds T_safe")

ax.set_title("System Architecture: Uncertainty-Aware CHO Timing", fontsize=11)
plt.tight_layout()
plt.savefig("figures/system_architecture.png", dpi=150)
print("Saved figures/system_architecture.png")