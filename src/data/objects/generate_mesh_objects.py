import argparse
import itertools
import pickle
import random
import sys
from distutils.util import strtobool
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from tqdm import trange

import src.data


def save_pickle(data, filename):
    # Ensure directory exists
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "wb") as f:
        pickle.dump(data, f)


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()


class WeightedRandomAgent:
    def __init__(self, action_space, rng, probs):
        self.action_space = action_space
        self.rng = rng
        self.probs = probs

    def act(self, observation):
        action = self.rng.choice(np.arange(self.action_space.n), p=self.probs)
        return action


def generate_dataset(args):
    # Set seed for numpy and random
    random.seed(args.seed)
    np.random.seed(args.seed)

    save_path = Path(args.save_path)

    # Cast args to correct types
    for k, v in args.env_kwargs.items():
        if k in ["width", "height", "num_nodes", "num_objects"]:
            args.env_kwargs[k] = int(v)
        if k in ["z_stddev"]:
            args.env_kwargs[k] = float(v)
    for k, v in args.reset_options.items():
        args.reset_options[k] = bool(strtobool(v))

    env = gym.make(args.env_id, **args.env_kwargs, disable_env_checker=True)
    env = TimeLimit(env.unwrapped, args.env_timelimit)

    # Seed environment once
    obs, _ = env.reset(seed=args.seed, options=args.reset_options)
    env.action_space.seed(args.seed)

    if args.env_id.startswith("MeshObjects"):
        probs = [0.1, 0.8, 0.1] * args.env_kwargs["num_objects"]
        probs = np.array(probs) / args.env_kwargs["num_objects"]
        agent = WeightedRandomAgent(env.action_space, env.np_random, probs)

    for i in trange(args.num_episodes, desc="Episode", file=sys.stdout):
        buffer = {key: [] for key in env.observation_space.keys()}
        buffer["reset_info"] = []
        buffer["action"] = []

        obs, reset_info = env.reset(options=args.reset_options)
        buffer["reset_info"].append(reset_info)
        buffer["pos"].append(obs["pos"])
        buffer["faces"].append(obs["faces"])

        terminated = truncated = False

        for _ in itertools.count():
            for k, v in obs.items():
                if k not in ["pos", "faces"]:
                    buffer[k].append(v)

            action = agent.act(obs)
            buffer["action"].append(action)

            obs, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

        # Save last state
        for k, v in obs.items():
            if k not in ["pos", "faces"]:
                buffer[k].append(v)

        for k, v in buffer.items():
            if k == "reset_info":
                buffer[k] = v[0]
            else:
                buffer[k] = np.asarray(v).squeeze()

        # Save episode
        save_pickle(buffer, save_path / f"episode{i+1}.h5")

    env.close()


if __name__ == "__main__":

    class ParseKwargs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, dict())
            for value in values:
                key, value = value.split("=")
                getattr(namespace, self.dest)[key] = value

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--env_id", type=str, help="Environment ID")
    parser.add_argument("--env_kwargs", nargs="*", action=ParseKwargs, default={})
    parser.add_argument("--reset_options", nargs="*", action=ParseKwargs, default={})
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Total number of episodes to simulate.",
    )
    parser.add_argument(
        "--env_timelimit",
        type=int,
        default=100,
        help="Max timelimit of env",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Save path for replay buffer (including extension .h5)",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    args = parser.parse_args()

    generate_dataset(args)
