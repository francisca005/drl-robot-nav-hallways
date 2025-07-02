import matplotlib, matplotlib.pyplot as plt
import os, sys, pandas as pd, glob

if len(sys.argv) != 2:
    print("Usage: python plot_trajectory.py <episode_id>")
    sys.exit(1)

episode_id = sys.argv[1]
dirname = f"data/positions/{episode_id}/"
out = f"figures/trajectories/t_{episode_id}.png"

matplotlib.use("Agg")

fig, ax = plt.subplots()

csv_files = glob.glob(os.path.join(dirname, "*.csv"))

if not csv_files:
    print(f"No csv files found in {dirname}")
    exit(1)

for path in csv_files:
    try:
        df = pd.read_csv(path)
        if not {"x", "y"}.issubset(df.columns):
            print(f"Skipping {path}: missing 'x' or 'y' columns")
            continue

        ax.plot(df["x"], df["y"], linewidth=2, color="red")

        # Draw final point as a small circle
        x_last, y_last = df["x"].iloc[-1], df["y"].iloc[-1]
        ax.plot(x_last, y_last, "o", markersize=5, color="red")

    except Exception as e:
        print(f"Failed to process {path}: {e}")

ax.set_axis_off()
plt.margins(0)
ax.set_aspect("equal", adjustable="box")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.savefig(out, bbox_inches="tight", pad_inches=0, transparent=True)
plt.close()

print(f"Saved trajectory overlay image: {out}")
