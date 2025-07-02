import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: python plot_trajectory.py <episode_id>")
    sys.exit(1)

episode_id = sys.argv[1]
filename = f"trajectory_{episode_id}.csv"

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"File not found: {filename}")
    sys.exit(1)

# Plot x vs z (horizontal plane)
plt.plot(df["x"], df["z"], marker="o", markersize=2, linewidth=1)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title(f"Trajectory for Episode {episode_id}")
plt.grid(True)
plt.axis("equal")
plt.savefig(f"trajectory_{episode_id}.pdf", bbox_inches="tight")
plt.show()
