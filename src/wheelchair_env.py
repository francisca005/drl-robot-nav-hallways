from robot_state import RobotState
from typing import Tuple

import gymnasium as gym
import numpy as np
import zmq
from gymnasium.spaces import Box

#criar um ambiente Gymnasium
class WheelchairEnv(gym.Env):
    def __init__(self, env_id: int):
        super(WheelchairEnv, self).__init__()

        context = zmq.Context() #ligação com robt_client.py
        self.socket = context.socket(zmq.REQ) #pedidos
        self.socket.bind(f"tcp://127.0.0.1:{5550 + env_id}")

        self.env_id = env_id

        #6 ações discretas
        self.action_space = gym.spaces.Discrete(6)

        v, w = 1, 3 # 1 lienar,3 angular
        self.to_action = lambda x: (
            [
                [v, 0],    # Forward
                [v, w],    # Forward and left
                [v, -w],   # Forward and right
                [0, w],    # Left
                [0, -w],   # Right
                [0, 0],    # Stop
            ]
        )[x]

        #360 LiDAR (de 0 a 10) + 1 valor da ação anterior (de 0 a 5)
        self.observation_space = Box(
            low=np.concatenate([np.full(360, 0.0), [0]]),
            high=np.concatenate([np.full(360, 10.0), [5]]),
            dtype=np.float64,
        )
        self.obs_shape = self.observation_space.shape

        self.no_obs()
        self.prev_action = 0 #ação anterior a 0
        self.prev_pref = 0.0 #preferência anterior de navegação
        self.time_step = 0 #steps desde o início
        self.time_limit = 5_000 #max por episódio
        self.commitment_threshold = 2.5 #reward de navegação

        self.timeout_reset_command = np.array([-999.0, -999.0], dtype=np.float32)

    def step(self, action: int):
        self.prev_action = action #ação escolhida
        action_values = self.to_action(action) #passar a velocidades

        reward = self.get_reward(self.prev_lidar, action_values) #reward = estado anterior + ação escolhida

        obs = self.send_action_get_obs(action_values) #envia ação robot_client.py

        if obs.collided:
            reward += self.collision_reward() #penalização de colisão, agora a -10
        elif obs.goal_reached:
            reward += self.goal_reward() #reward de objetivo, agora a 0

        self.prev_lidar = obs.lidar #atualiza LiDAR para o prox step
        self.time_step += 1

        terminated = obs.collided or obs.goal_reached #colisão ou objetivo
        truncated = self.time_step >= self.time_limit #tempo limite excedido

        if truncated and not terminated:
            reward -= 20 #se deu timeout é penalizado, aumentado para 20


        info = {
            "is_success": obs.goal_reached if not truncated else False,
            "collision": obs.collided if not truncated else False,
            "timeout": truncated and not terminated,
        }

        return obs.to_array(), reward, terminated, truncated, info

    #timeout lógica antiga
    def request_timeout_reset(self) -> RobotState:
        """
        Inform the Webots controller that the current episode ended by timeout.

        This allows the controller to:
            1. save the trajectory;
            2. reset the physical robot;
            3. return a fresh observation to keep the REQ/REP socket synchronized.
        """
        self.socket.send_pyobj(self.timeout_reset_command)
        return self.get_observation()

    def get_reward(self, obs: np.ndarray, action: Tuple[int, int]) -> float:
        v, w = action

        r_distance = 1.0 if v > 0 else 0.0 #andar em frente +1

        min_range = np.min(obs[140:220]) #raios frontais
        collision_threshold = 1.0

        if min_range < collision_threshold:
            r_collision = -np.exp(3 * (collision_threshold - min_range)) + 1 #penalização mais negativa quanto mais perto
        else:
            r_collision = 0.0

        r_navigation = self.navigation_reward(obs, action) #recompensa por direção adequada
        r_stability = self.stability_reward(obs, action) #penalização por virar

        r_stop = -0.5 if v == 0 and w == 0 else 0.0 #nova mudificação para penalizar quando fica parado

        return r_distance + r_collision + r_navigation + r_stability + r_stop

    def navigation_reward(self, obs: np.ndarray, action: Tuple[int, int]) -> float:
        _, w = action

        left_sector = obs[100:170]
        front_sector = obs[170:190]
        right_sector = obs[190:260]

        left_clearance = np.mean(left_sector) #dist média livre à esquerda
        right_clearance = np.mean(right_sector)

        #pelo menos uma leitura frontal a menos de 2.5
        obstacle_ahead = np.any(front_sector < self.commitment_threshold)

        if obstacle_ahead:
            # >0 direita mais livre, <0 esquerda mais livre
            clearance_diff = right_clearance - left_clearance

            #60% da preferência anterior, 40% da diferença atual
            alpha = 0.40
            self.prev_pref = (1 - alpha) * self.prev_pref + alpha * clearance_diff

            r = 3.0

            #pref direita mais livre
            if self.prev_pref > 0.2:
                return r if w < 0 else -r

            #pref esquerda mais livre
            if self.prev_pref < -0.2:
                return r if w > 0 else -r

            #espaço direita mais livre
            if clearance_diff > 0.2:
                return r if w < 0 else -r * 0.5
            
            #espaço esquerda mais livre
            if clearance_diff < -0.2:
                return r if w > 0 else -r * 0.5

            #perto e não vira
            if np.min(front_sector) < 1.0:
                return r if w != 0 else -r

        elif w == 0:
            return 1.0

        return 0

    def stability_reward(self, obs: np.ndarray, action: Tuple[int, int]) -> float:
        _, w = action

        #se vira pequena penalização
        if w != 0:
            return -0.2

        return 0

    def reset_preference(self):
        self.prev_pref = 0.0

    def collision_reward(self) -> int:
        return -10

    #modificação aumentar para 30
    def goal_reward(self) -> int:
        return 30

    def send_action_get_obs(self, action: Tuple[int, int]) -> RobotState:
        self.socket.send_pyobj(action) #eviar ação
        return self.get_observation() #esperar resposta

    def get_observation(self) -> RobotState:
        state = self.socket.recv_pyobj()
        state.prev_action = self.prev_action
        return state

    def no_obs(self) -> np.ndarray:
        return np.zeros(self.obs_shape, dtype=np.float64)

    def reset(self, seed: int = None, options=None) -> Tuple[np.ndarray, dict]:
        obs = self.no_obs()
        self.prev_lidar = obs[:360]
        self.time_step = 0
        self.reset_preference()
        return obs, {}

    def close(self):
        print("Closing environment " + str(self.env_id))

        try:
            self.socket.send_pyobj([-1])
        except Exception:
            pass

        self.socket.close()
        zmq.Context.instance().destroy()

        return super().close()