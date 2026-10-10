import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(6, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 15)
ax.axis("off")

def box(x, y, w, h, text, color="#dbe9f6", fontsize=9):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                        facecolor=color, edgecolor="black", linewidth=1)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)

def down_arrow(x, y_top, y_bottom):
    a = FancyArrowPatch((x, y_top), (x, y_bottom), arrowstyle="-|>",
                         mutation_scale=16, color="black", linewidth=1.3)
    ax.add_patch(a)

# Single top-to-bottom pipeline, one box per row, no branching
box(1, 13.2, 8, 1.1, "TLE orbital data\n(CelesTrak, archived)")
down_arrow(5, 13.2, 12.5)

box(1, 11.4, 8, 1.1, "Orbital propagation (SGP4)\nelevation, slant range vs. time\nGround station: Aberdeen, Scotland")
down_arrow(5, 11.4, 10.7)

box(1, 9.6, 8, 1.1, "Link budget model\nCNR(t) per 3GPP TR 38.821")
down_arrow(5, 9.6, 8.9)

box(1, 7.8, 8, 1.1, "Remaining usable link time\nestimate: T_link_hat(t)")
down_arrow(5, 7.8, 7.1)

box(1, 6.0, 8, 1.1, "Uncertainty model: sigma_T(t)\n(link-margin, delay, geometry terms)")
down_arrow(5, 6.0, 5.3)

box(1, 4.0, 8, 1.3, "Trigger policy\nT_safe(t) = T_link_hat(t) - z_alpha * sigma_T(t)\nTrigger when T_safe(t) <= T_overhead",
    color="#fce8d5")
down_arrow(5, 4.0, 3.3)

box(1, 2.0, 8, 1.1, "Handover decision:\ntrigger CHO execution to target satellite")

ax.set_title("System Pipeline: Uncertainty-Aware CHO Timing", fontsize=11, pad=15)
plt.tight_layout()
plt.savefig("figures/system_architecture.png", dpi=150, bbox_inches="tight")
print("Saved figures/system_architecture.png")