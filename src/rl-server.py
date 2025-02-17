import socket
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from wheelchair_env import WheelchairEnv

HOST = "127.0.0.1"
PORT = 5000
NUM_ROBOTS = 2
EPISODES = 100
TRAIN_STEPS = 1000


def train_model():
    """Starts PPO training with Webots vectorized environments."""
    env = make_vec_env(WheelchairEnv, n_envs=NUM_ROBOTS, vec_env_cls=SubprocVecEnv)

    model = PPO("MlpPolicy", env, verbose=1)

    # Start socket server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(NUM_ROBOTS)
    print(f"Server listening on {HOST}:{PORT}")

    connections = []
    for _ in range(NUM_ROBOTS):
        conn, addr = server.accept()
        connections.append(conn)
        print(f"Connection accepted from {addr}")

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

        model.learn(total_timesteps=10)

    server.close()


if __name__ == "__main__":
    train_model()
