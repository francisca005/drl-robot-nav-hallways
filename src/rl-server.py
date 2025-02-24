import sys
import socket
import pickle
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.buffers import RolloutBuffer
from wheelchair_env import WheelchairEnv

HOST = "127.0.0.1"
PORT = 5000
N_ROBOTS = 2
EPISODES = 2000
TRAIN_STEPS = 1000


def train_model():
    """Start server"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(N_ROBOTS)
    print(f"Server listening on {HOST}:{PORT}")

    connections = []
    for _ in range(N_ROBOTS):
        conn, addr = server.accept()
        connections.append(conn)
        print(f"Connection accepted from {addr}")

    """Starts PPO training with Webots vectorized environments."""
    env = make_vec_env(
        WheelchairEnv,
        n_envs=N_ROBOTS,
        vec_env_cls=SubprocVecEnv,
    )
    check_env(env)

    model = PPO("MlpPolicy", env, verbose=1)

    # 3️⃣ Initialize a rollout buffer
    buffer_size = 2048  # Usually set to the same as 'n_steps' in PPO
    rollout_buffer = RolloutBuffer(
        buffer_size=buffer_size,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device="cpu",
        gae_lambda=0.95,
        gamma=0.99,
        n_envs=N_ROBOTS,
    )

    # 4️⃣ Custom Training Loop
    TOTAL_TIMESTEPS = 100000
    batch_size = 64
    timesteps_collected = 0

    obs = env.reset()
    while timesteps_collected < TOTAL_TIMESTEPS:
        rollout_buffer.reset()

        for _ in range(buffer_size):
            # 5️⃣ Predict actions using current policy
            with torch.no_grad():
                actions, values, log_probs = model.policy(obs, deterministic=False)

            # 6️⃣ Step environment
            next_obs, rewards, dones, infos = env.step(actions.cpu().numpy())

            # 7️⃣ Store in rollout buffer
            rollout_buffer.add(obs, actions, rewards, dones, values, log_probs)

            obs = next_obs
            timesteps_collected += NUM_ENVS

        # 8️⃣ Compute advantages and return
        with torch.no_grad():
            last_values = model.policy.predict_values(obs)
        rollout_buffer.compute_returns_and_advantage(last_values, dones)

        # 9️⃣ Train the model using the collected batch
        model.train()  # This optimizes using the collected rollout buffer

        print(f"Timesteps Collected: {timesteps_collected}/{TOTAL_TIMESTEPS}")

    print("Training complete!")

    model = PPO("MlpPolicy", env, verbose=1)

    for _ in range(EPISODES):
        observations = []

        """ Collect observations from robots """
        for conn in connections:
            data = conn.recv(4096)
            obs = pickle.loads(data)
            observations.append(obs)

        """ Predict actions with model """
        actions, _ = model.predict(observations, deterministic=False)

        """ Send actions to robots """
        for conn, action in zip(connections, actions):
            conn.sendall(pickle.dumps(action))

        model.learn(total_timesteps=TRAIN_STEPS)

    server.close()


if __name__ == "__main__":
    train_model()
