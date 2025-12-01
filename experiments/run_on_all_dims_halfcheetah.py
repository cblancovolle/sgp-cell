import subprocess
import sys

base_cmd = [
    sys.executable,
    r".\experiments\sgpcell_minari.py",
    "--dataset_id",
    "mujoco/halfcheetah/medium-v0",
    "--eval_freq",
    "1000",
    "--train_size",
    "100",
    "--test_size",
    "20",
    "--min_points",
    "15",
    "--reconstruct_th",
    "0.2",
    "--k_components",
    "4",
    "--confidence_forget",
    "0.1",
    "--confidence_steepness",
    "3.0",
    "--destroy_th",
    "0.1",
    "--neighbor_confidence",
    "0.99",
    "--aic_param_coeff",
    "0.1",
]

for out_dim in range(0, 11):  # 0 to 10 inclusive
    name = f"halfcheetah_{out_dim}"
    cmd = base_cmd + ["--name", name, "--out_dim", str(out_dim)]

    print(f"\nRunning: {cmd}")
    subprocess.run(cmd, check=True)
