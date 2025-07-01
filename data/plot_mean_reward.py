from tbparse import SummaryReader
import matplotlib.pyplot as plt

# Adjust path to your event files
reader = SummaryReader("logs/ppo.log", pivot=True)
df = reader.scalars

# Filter to one tag (e.g., rollout/ep_rew_mean)
rew_df = df[df["tag"] == "rollout/ep_rew_mean"]

# Plot
plt.plot(rew_df["step"], rew_df["value"])
plt.xlabel("Timesteps")
plt.ylabel("Mean Episode Reward")
plt.title("Training Curve: PPO")
plt.grid(True)
plt.savefig("reward_plot.pdf", bbox_inches="tight")  # High-res export
plt.show()
