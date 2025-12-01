import os
import sys

sys.path.append(".")

import torch
import tqdm
import minari
import json
import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from minari import MinariDataset
from sgp_cell.trainers import PCAOnlineTrainer
from sgp_cell.agents import MultiOutputSmtAgent
from pathlib import Path
from sklearn.metrics import (
    root_mean_squared_error,
    mean_squared_error,
    mean_absolute_error,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run your experiment with specified parameters."
    )

    # Required parameter
    parser.add_argument(
        "--name", type=str, required=True, help="Name of the experiment"
    )

    parser.add_argument(
        "--log_snapshot",
        type=bool,
        default=True,
        help="Whether to log the snapshots of the trainer or not.",
    )

    # Optional parameters with defaults
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="mujoco/hopper/medium-v0",
        help="ID of the dataset to use for training",
    )
    parser.add_argument(
        "--train_size", type=int, default=100, help="Number of training episodes"
    )
    parser.add_argument(
        "--test_size", type=int, default=20, help="Number of test episodes"
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=1000,
        help="Evaluation frequency in number of steps",
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=None,
        help="Min number of points to instanciate agent",
    )
    parser.add_argument(
        "--corr",
        type=str,
        choices=["abs_exp", "squar_exp", "matern32", "matern52"],
        default="squar_exp",
        help="Kernel type to use in agents",
    )
    parser.add_argument(
        "--poly",
        type=str,
        choices=["constant", "linear", "quadratic"],
        default="constant",
        help="Regression function type",
    )
    parser.add_argument(
        "--k_components",
        type=int,
        default=3,
        help="Number of components for PCA in agents",
    )
    parser.add_argument(
        "--reconstruct_th",
        type=float,
        default=0.2,
        help="Reconstruction error threshold for PCA (0.0-1.0)",
    )
    parser.add_argument(
        "--neighbor_confidence",
        type=float,
        default=0.95,
        help="Neighborhood confidence threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--confidence_steepness",
        type=float,
        default=2.0,
        help="Steepness of agent confidence curve",
    )
    parser.add_argument(
        "--confidence_forget",
        type=float,
        default=0.2,
        help="Confidence forget lambda factor (0.0-1.0)",
    )
    parser.add_argument(
        "--destroy_th",
        type=float,
        default=0.2,
        help="Confidence forget destroy threshold",
    )
    parser.add_argument(
        "--aic_param_coeff",
        type=float,
        default=0.1,
        help="Influence of parameters in aic objective in agent updates",
    )
    parser.add_argument(
        "--out_dim",
        type=int,
        default=slice(None),
        help="Index of the dim to model in state vector",
    )

    args = parser.parse_args()
    return args


args = parse_args()

run_id = f"{args.name}_{int(time.time())}"
log_path = Path(f"logs/{run_id}/training_log.jsonl")
snapshot_path = Path(f"logs/{run_id}/snapshots/")
eval_log_path = Path(f"logs/{run_id}/training_eval_log.jsonl")


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize(x) for x in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def log_entry(log_path: Path, entry: dict):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = sanitize(entry)
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())


def log_snapshot(log_path: Path, entry: dict, step: int):
    path = log_path / f"snapshot_{step}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(entry, f)


def evaluate(trainer: PCAOnlineTrainer, dataset: MinariDataset):
    y_preds = []
    y_test = []
    with tqdm.trange(len(dataset)) as pbar:
        for ep_id in pbar:
            ep = dataset[ep_id]
            for i in range(len(ep) - 1):
                obs, action, next_obs = (
                    ep.observations[i],
                    ep.actions[i],
                    ep.observations[i + 1],
                )
                x_new = torch.as_tensor(np.hstack([obs, action])).view(1, -1)
                y_new = torch.as_tensor(next_obs[args.out_dim]).view(1, -1)

                if trainer.n_agents > 0:
                    with torch.no_grad():
                        mu, var = trainer.predict_one(x_new)
                else:
                    mu = torch.full_like(y_new, torch.nan)

                y_preds += [mu.numpy()]
                y_test += [y_new.numpy()]
    y_preds = np.vstack(y_preds)
    y_test = np.vstack(y_test)

    try:
        rmse = root_mean_squared_error(y_preds, y_test)
        mse = mean_squared_error(y_preds, y_test)
        mae = mean_absolute_error(y_preds, y_test)
    except ValueError:
        rmse = np.nan
        mse = np.nan
        mae = np.nan
    return {"rmse": rmse, "mse": mse, "mae": mae}


# ====== Dataset Import ======
dataset_id = args.dataset_id
dataset = minari.load_dataset(dataset_id)
train_dataset, test_dataset = minari.split_dataset(
    dataset, sizes=[args.train_size, args.test_size], seed=42
)
print("Observation space:", dataset.observation_space)
print("Action space:", dataset.action_space)
print("Total episodes:", dataset.total_episodes)
print("Total steps:", dataset.total_steps)

state_dim = dataset.observation_space.shape[0]
action_dim = dataset.action_space.shape[0]

# ====== Instanciate Trainer ======
trainer = PCAOnlineTrainer(
    in_dim=state_dim + action_dim,
    out_dim=1 if isinstance(args.out_dim, int) else state_dim,
    agent_cls=MultiOutputSmtAgent,
    neighbor_confidence=args.neighbor_confidence,
    min_points=state_dim + action_dim + 1 if not args.min_points else args.min_points,
    k_components=args.k_components,
    reconstruct_threshold=args.reconstruct_th,
    confidence_norm_steepness=args.confidence_steepness,
    confidence_forget_lambd=args.confidence_forget,
    confidence_destroy_th=args.destroy_th,
    agent_kwargs=dict(
        model_kwargs=dict(
            poly=args.poly,
            corr=args.corr,
        ),
        param_coeff=args.aic_param_coeff,
    ),
)

# ====== Training Loop ======
step = 0
eval_freq = args.eval_freq
for ep_id, ep in enumerate(train_dataset):
    trainer.reset()
    with tqdm.trange(len(ep) - 1) as pbar:
        for i in pbar:
            obs, action, next_obs = (
                ep.observations[i],
                ep.actions[i],
                ep.observations[i + 1],
            )
            x_new = torch.as_tensor(np.hstack([obs, action])).view(1, -1)
            y_new = torch.as_tensor(next_obs[args.out_dim]).view(1, -1)

            if trainer.n_agents > 0:
                with torch.no_grad():
                    mu, var = trainer.predict_one(x_new)
            else:
                mu = torch.full_like(y_new, torch.nan)

            err = np.abs((mu - y_new).mean().numpy())

            infos = trainer.learn_one(x_new, y_new)
            infos["abs_err"] = err
            infos["episode_id"] = ep_id
            infos["step"] = step
            if isinstance(args.out_dim, int):
                infos["out_dim_id"] = args.out_dim
            log_entry(log_path, infos)

            pbar.set_description(f"[Ep {ep_id} - Agents {trainer.n_agents}]")
            if step % eval_freq == 500:
                eval_infos = evaluate(trainer, test_dataset)
                eval_infos["step"] = step
                eval_infos["n_retained"] = np.sum(
                    [a.current_mem_size for a in trainer.agents]
                )
                log_entry(eval_log_path, eval_infos)
                if args.log_snapshot:
                    log_snapshot(snapshot_path, trainer.snapshot(), step)
            step += 1
