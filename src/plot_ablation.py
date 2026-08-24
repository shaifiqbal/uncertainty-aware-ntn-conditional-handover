import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/ablation_study.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

labels = ["Full\nproposed", "No uncertainty\nmargin", "No delay\nterm"]
colors = ["tab:blue", "tab:red", "tab:orange"]

ax1.bar(labels, df["failure_probability"] * 100, color=colors)
for i, v in enumerate(df["failure_probability"] * 100):
    ax1.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8)
ax1.set_ylabel("Handover failure probability (%)")
ax1.set_title("Ablation: reliability")

ax2.bar(labels, df["avg_excess_margin_s"], color=colors)
for i, v in enumerate(df["avg_excess_margin_s"]):
    ax2.text(i, v + 1, f"{v:.1f}s", ha="center", fontsize=8)
ax2.set_ylabel("Avg. excess margin (s)")
ax2.set_title("Ablation: efficiency cost")

plt.tight_layout()
plt.savefig("figures/ablation_study.png", dpi=150)
print("Saved figures/ablation_study.png")