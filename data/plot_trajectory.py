import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: python plot_trajectory.py <episode_id>")
    sys.exit(1)

episode_id = sys.argv[1]
filename = f"data/positions/trajectory_{episode_id}.csv"
out = f"figures/trajectories/trajectory_{episode_id}.png"

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"File not found: {filename}")
    sys.exit(1)

fig, ax = plt.subplots()
ax.plot(df["x"], df["y"], color="red", linewidth=2)  # or customize color

ax.set_axis_off()
plt.margins(0)
plt.gca().set_aspect("equal", adjustable="box")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.savefig(
    out,
    bbox_inches="tight",
    pad_inches=0,
    transparent=True,
)
plt.close()

print(f"Saved trajectory image at {out}")
