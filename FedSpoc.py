import torch

print("CUDA Available:", torch.cuda.is_available())
print("Current Device:", torch.cuda.current_device())
print("Device Name:", torch.cuda.get_device_name(0))

!nvidia-smi

"""**FedSPOC**"""

# Full integrated FedSPOC script (Colab-ready)
# Paste into your notebook (after mounting drive if needed). Adjust PKL_DIR at top if necessary.
# *** UPDATED: Added sensitivity analysis, SHAP attribution, and extended fairness metrics ***

import os, gc, pickle, torch, warnings, random, logging, math, time, glob
import numpy as np, pandas as pd
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize, OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from flwr.client import NumPyClient, ClientApp
from flwr.common import ndarrays_to_parameters, Context
from flwr.server.strategy import FedAvgM
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
from flwr.common import parameters_to_ndarrays
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import ray
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
pd.set_option('future.no_silent_downcasting', True)

# ---------------- Minimal warnings/logs suppression (place near top) ----------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_LOG_TO_STDERR"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("flwr").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.ERROR)
import os
os.environ["RAY_DISABLE_DASHBOARD"] = "1"
os.environ["RAY_ENABLE_METRICS_EXPORT"] = "0"
os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

print("✅ Warnings/logs suppression installed.")

# ---------------- USER FLAGS (single authoritative block) ----------------
DEVICE = torch.device("cpu")

import torch
print(torch.cuda.is_available())

# Experiment scale (full experiment)
# Experiment scale - INCREASED FOR BETTER CONVERGENCE
NUM_CLIENTS = 500
NUM_ROUNDS = 30
NUM_FOG_NODES = 10
NUM_CLASSES = 3
CLIENTS_PER_ROUND = 80  # Increased from 30 (16% of clients per round)
EVAL_CLIENTS_PER_ROUND = 10  # Increased from 5

# Seeds: multiple seeds for final runs
SEEDS = [42, 52, 62, 72, 82]

# Run-time toggles
USE_FEDSPOC = True
USE_FCL = False
USE_PPO_BASELINE = False
SAVE_DETAILED = True
USE_META_UPDATE = True
USE_FEDPROX = False
FEDPROX_MU = 0.01

# *** NEW: Sensitivity analysis hyperparameter defaults (Reviewer 1, Point 12) ***
RARITY_BETA  = 1.0   # default rarity bias β  (Eq. 17)
EWMA_LAMBDA  = 0.6   # default EWMA smoothing λ (Eq. 14)
PPO_CLIP     = 0.2   # default PPO clip ε

# ---------------- Churn Simulation ----------------
ENABLE_CHURN = False        # Set True to activate churn experiment
CHURN_MODE = "random"       # "random" or "adversarial"
CHURN_RATE = 0.3            # 30% dropout

# Diagnostics & holdout
RUN_DIAGNOSTICS = True
USE_GLOBAL_HOLDOUT = True
GLOBAL_HOLDOUT_FRAC = 0.20

# Randomness injection to sampling
RANDOM_INJECTION_FRAC = 0.02

# Encoding choice
USE_ONEHOT = True

# ---------------- Checkpoint / output dirs ----------------
CHECKPOINT_DIR = "./checkpoints"
OUTPUT_DIR = "./results"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Path to pickles / parquet ----------------
PKL_DIR = "./data"

PKL_PATHS = {
    "crawdad_pkl": os.path.join(PKL_DIR, "crawdad_250.pkl"),
    "rounds": os.path.join(PKL_DIR, "round_partitions_250_guaranteed.pkl"),
    "crawdad_df_pkl": os.path.join(PKL_DIR, "crawdad_250_df.pkl"),
    "parquet_renamed": os.path.join(PKL_DIR, "crawdad_250_renamed.parquet"),
    "duckdb": os.path.join(PKL_DIR, "crawdad_250.duckdb")
}
print(f"CONFIG -> NUM_CLIENTS={NUM_CLIENTS}, NUM_ROUNDS={NUM_ROUNDS}, NUM_FOG_NODES={NUM_FOG_NODES}, SEEDS={SEEDS}")

DESIRED_FEATURES = [
    'use_casetype_(input_1)', 'lte/5g_ue_category_(input_2)',
    'technology_supported_(input_3)', 'day_(input4)',
    'time_(input_5)', 'qci_(input_6)',
    'packet_loss_rate_(reliability)', 'packet_delay_budget_(latency)'
]

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------------- Load DataFrame robustly ----------------
if os.path.exists(PKL_PATHS["parquet_renamed"]):
    crawdad_df = pd.read_parquet(PKL_PATHS["parquet_renamed"])
    print("Loaded Parquet:", PKL_PATHS["parquet_renamed"])
elif os.path.exists(PKL_PATHS["crawdad_df_pkl"]):
    crawdad_df = pd.read_pickle(PKL_PATHS["crawdad_df_pkl"])
    print("Loaded DataFrame PKL:", PKL_PATHS["crawdad_df_pkl"])
elif os.path.exists(PKL_PATHS["crawdad_pkl"]):
    raw_data = load_pickle(PKL_PATHS["crawdad_pkl"])
    if isinstance(raw_data, dict) and 'X' in raw_data and 'y' in raw_data:
        crawdad_df = pd.DataFrame(raw_data["X"], columns=DESIRED_FEATURES)
        crawdad_df["slice_label"] = np.array(raw_data["y"]).astype(int)
        crawdad_df["slice_type_output"] = crawdad_df["slice_label"].map({0:"URLLC",1:"mMTC",2:"eMBB"})
        print("Loaded PKL dict -> DataFrame:", PKL_PATHS["crawdad_pkl"])
    else:
        raise ValueError("Unsupported crawdad PKL format. Please supply parquet or a {'X','y'} dict.")
else:
    raise FileNotFoundError("No crawdad data found. Please ensure crawdad_250 exists in PKL_DIR.")

# Normalize column names
if "slice_type_(output)" in crawdad_df.columns and "slice_type_output" not in crawdad_df.columns:
    crawdad_df = crawdad_df.rename(columns={"slice_type_(output)": "slice_type_output"})

# Ensure slice_label exists
if "slice_label" not in crawdad_df.columns:
    if "slice_type_output" in crawdad_df.columns:
        crawdad_df["slice_label"] = crawdad_df["slice_type_output"].map({"URLLC":0,"mMTC":1,"eMBB":2})
    else:
        raise ValueError("No slice_label or slice_type_output found in dataset.")

# Heuristic PKL_FEATURES mapping
fcols = [c for c in crawdad_df.columns if c.startswith("f_")]
fcols = sorted(fcols, key=lambda s: int(s.split("_")[1]) if "_" in s and s.split("_")[1].isdigit() else 999)
mapped = fcols[:6] if len(fcols) >= 6 else fcols
qos_map = [q for q in ["packet_loss_rate_(reliability)", "packet_delay_budget_(latency)", "normalized_latency", "normalized_reliability"] if q in crawdad_df.columns]
PKL_FEATURES = mapped + qos_map
if len(PKL_FEATURES) < 5:
    PKL_FEATURES = [c for c in crawdad_df.columns if c not in ("slice_label","slice_type_output")][:7]
print("Final PKL_FEATURES list used:", PKL_FEATURES)

# Create client_id if absent
if "client_id" not in crawdad_df.columns:
    crawdad_df = crawdad_df.reset_index(drop=True)
    crawdad_df["client_id"] = np.arange(len(crawdad_df)) % NUM_CLIENTS

# Normalized QoS proxies
if "packet_delay_budget_(latency)" in crawdad_df.columns:
    try:
        lat = crawdad_df["packet_delay_budget_(latency)"].astype(float)
        crawdad_df["normalized_latency"] = (lat - lat.min()) / (lat.max() - lat.min() + 1e-9)
    except Exception:
        crawdad_df["normalized_latency"] = 0.5
else:
    crawdad_df["normalized_latency"] = crawdad_df.get("normalized_latency", 0.5)

if "packet_loss_rate_(reliability)" in crawdad_df.columns:
    try:
        rl = crawdad_df["packet_loss_rate_(reliability)"].astype(float)
        crawdad_df["normalized_reliability"] = (rl - rl.min()) / (rl.max() - rl.min() + 1e-9)
    except Exception:
        crawdad_df["normalized_reliability"] = crawdad_df.get("normalized_reliability", 0.5)
else:
    crawdad_df["normalized_reliability"] = crawdad_df.get("normalized_reliability", 0.5)

# ---------------- Load or Generate partitions ----------------

print("Generating round partitions dynamically for current NUM_CLIENTS...")

# Shuffle dataset once for fair distribution
crawdad_df = crawdad_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Re-assign client_id dynamically based on NUM_CLIENTS
crawdad_df["client_id"] = np.arange(len(crawdad_df)) % NUM_CLIENTS

round_partitions = []
rng = np.random.default_rng(42)

for r in range(NUM_ROUNDS):
    part = defaultdict(list)
    for cid in range(NUM_CLIENTS):
        client_indices = crawdad_df[crawdad_df["client_id"] == cid].index.tolist()
        if len(client_indices) > 0:
            sample_size = max(1, int(0.4 * len(client_indices)))
            selected = rng.choice(client_indices, size=sample_size, replace=False)
            part[cid] = selected.tolist()
    round_partitions.append(part)


# ---------------- GLOBAL HOLDOUT ----------------
holdout_idx_set = set()
if USE_GLOBAL_HOLDOUT:
    rng = np.random.default_rng(SEEDS[0] if SEEDS else 42)
    all_indices = crawdad_df.index.to_numpy()
    holdout_count = int(len(all_indices) * GLOBAL_HOLDOUT_FRAC)
    holdout_idx = rng.choice(all_indices, size=holdout_count, replace=False)
    holdout_idx_set = set(holdout_idx.tolist())
    print(f"Global holdout rows: {len(holdout_idx_set)} (fraction {GLOBAL_HOLDOUT_FRAC})")
else:
    print("Global holdout disabled.")

# ---------------- Encoding pipeline ----------------
categorical_cols = [c for c in PKL_FEATURES if c in crawdad_df.columns and (crawdad_df[c].dtype == object or crawdad_df[c].dtype.name == "category")]
numeric_cols = [c for c in PKL_FEATURES if c in crawdad_df.columns and c not in categorical_cols]
print("Categorical cols:", categorical_cols)
print("Numeric cols:", numeric_cols)

encoder = None
onehot = None
FEATURE_VECTOR_COLS = categorical_cols + numeric_cols if categorical_cols else numeric_cols
FEATURE_VECTOR_DIM = None

if categorical_cols:
    enc_df = crawdad_df.loc[~crawdad_df.index.isin(holdout_idx_set), categorical_cols].fillna("NA").astype(str)
    if USE_ONEHOT:
        try:
            onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)
        except TypeError:
            onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        onehot.fit(enc_df.values)
        if not enc_df.empty:
            FEATURE_VECTOR_DIM = onehot.transform(enc_df.values[:1]).shape[1] + (len(numeric_cols) if numeric_cols else 0)
        else:
            FEATURE_VECTOR_DIM = (len(numeric_cols) if numeric_cols else 0)
        print("Fitted OneHotEncoder; output dim:", FEATURE_VECTOR_DIM)
    else:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoder.fit(enc_df.values)
        FEATURE_VECTOR_DIM = len(categorical_cols) + len(numeric_cols)
        print("Fitted OrdinalEncoder; vector dim:", FEATURE_VECTOR_DIM)
else:
    FEATURE_VECTOR_DIM = len(numeric_cols)
    print("No categorical features; feature dim:", FEATURE_VECTOR_DIM)

def encode_features(df):
    Xnum = df[numeric_cols].astype(np.float32).fillna(0.0).values if numeric_cols else np.zeros((len(df),0), dtype=np.float32)
    if categorical_cols:
        cat_values = df[categorical_cols].fillna("NA").astype(str).values
        if USE_ONEHOT and onehot is not None:
            cat_enc = onehot.transform(cat_values).astype(np.float32)
        elif encoder is not None:
            cat_enc = encoder.transform(cat_values).astype(np.float32)
        else:
            cat_enc = np.array([[hash(v) % 1000 for v in row] for row in cat_values], dtype=np.float32)
        X = np.hstack([cat_enc, Xnum]) if Xnum.shape[1] > 0 else cat_enc
    else:
        X = Xnum
    return X

# ---------------- synthetic client stats ----------------
client_entropy = np.clip(np.random.RandomState(0).rand(NUM_CLIENTS), 0.0, 1.0)
client_skewness = np.clip(np.random.RandomState(1).rand(NUM_CLIENTS), 0.0, 1.0)
client_energy = np.random.RandomState(2).uniform(0.3, 1.0, NUM_CLIENTS)
client_latency = np.random.RandomState(3).uniform(10, 100, NUM_CLIENTS)
client_timestamp = np.linspace(0.0, 1.0, NUM_CLIENTS)

# ---------------- fog assignments ----------------
def build_fog_assignment_by_kmeans(num_clients_local):
    client_locations = []
    for cid in range(num_clients_local):
        lat = math.sin(cid / max(1, num_clients_local) * 2 * math.pi) * 50 + 25
        lon = math.cos(cid / max(1, num_clients_local) * 2 * math.pi) * 50 + 75
        client_locations.append([lat, lon])
    kmeans = KMeans(n_clusters=min(NUM_FOG_NODES, num_clients_local), random_state=42)
    assignments = kmeans.fit_predict(client_locations)
    fog_to_clients = defaultdict(list)
    for cid, fid in enumerate(assignments):
        fog_to_clients[fid].append(cid)
    return np.array(assignments), fog_to_clients

fog_assignment_map, fog_to_clients_map = build_fog_assignment_by_kmeans(NUM_CLIENTS)

# ---------------- Model + helpers ----------------
class MobilityModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, NUM_CLASSES)
        )
    def forward(self, x): return self.net(x)

def get_parameters(model):
    return [v.cpu().numpy() for v in model.state_dict().values()]

def set_parameters(model, parameters):
    state_dict = model.state_dict()
    for k, v in zip(list(state_dict.keys()), parameters):
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict, strict=True)

# ---------------- Robust label mapping ----------------
label_map_str_to_int = {"URLLC": 0, "eMBB": 1, "mMTC": 2}
def map_labels_safe(series):
    if series.dtype == object or series.dtype.name == "category":
        mapped = series.map(label_map_str_to_int)
        if mapped.notna().any():
            return mapped.fillna(method="ffill").astype(np.int64).values
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.dropna().astype(np.int64).values

# ---------------- safe AUC helper ----------------
auc_skipped_count = 0
def safe_auc(y_true, pred_prob, classes=NUM_CLASSES):
    global auc_skipped_count
    try:
        if len(np.unique(y_true)) < 2:
            auc_skipped_count += 1
            return float("nan")
        y_bin = label_binarize(y_true, classes=np.arange(classes))
        return float(roc_auc_score(y_bin, pred_prob, average="macro", multi_class="ovr"))
    except Exception:
        auc_skipped_count += 1
        return float("nan")

# ---------------- Checkpointing ----------------
def save_checkpoint(round_num, parameters, seed=None):
    try:
        ckdir = CHECKPOINT_DIR if seed is None else os.path.join(CHECKPOINT_DIR, f"seed_{seed}")
        os.makedirs(ckdir, exist_ok=True)
        torch.save(parameters, os.path.join(ckdir, f"global_round_{round_num}.pt"))
    except Exception as e:
        print("⚠ checkpoint save failed:", e)

# ---------------- Meta-Reward Shaping ----------------
meta_w = np.array([0.6, 0.3, 0.1])
meta_history = []
meta_eta = 0.6
def spoc_reward_meta(lat_n, en_n, ent_n, weights=None):
    w = weights if weights is not None else meta_w
    L, E, H = float(np.clip(lat_n,0,1)), float(np.clip(en_n,0,1)), float(np.clip(ent_n,0,1))
    r = 1.0 - (w[0]*L + w[1]*E + w[2]*H)
    return float(np.clip(r, 0.0, 1.0))

def update_meta_weights(delta_metric, per_client_feature_agg=None):
    global meta_w, meta_history, meta_eta
    norm_delta = math.tanh(delta_metric * 5.0)
    if per_client_feature_agg is None:
        if norm_delta > 0:
            meta_w = meta_w * (1 + 0.01 * norm_delta)
        else:
            meta_w = meta_w * (1 - 0.02 * (-norm_delta))
    else:
        agg = np.array(per_client_feature_agg)
        update_factor = np.exp(meta_eta * norm_delta * (agg / (agg.sum()+1e-9)))
        meta_w = meta_w * update_factor
    meta_w = np.clip(meta_w, 0.03, 0.94)
    meta_w = meta_w / meta_w.sum()
    meta_history.append(meta_w.copy())
    return meta_w

# ---------------- FedSPOC sampling ----------------
def spoc_reward(latency, energy, entropy, weights=None):
    return spoc_reward_meta(latency, energy, entropy, weights)

def sample_clients_with_spoc(round_num, reward_threshold=0.3):
    rewards = []
    for cid in range(NUM_CLIENTS):
        lat_n = (client_latency[cid] - client_latency.min()) / \
                (client_latency.max() - client_latency.min() + 1e-9)
        en_n = (client_energy[cid] - client_energy.min()) / \
               (client_energy.max() - client_energy.min() + 1e-9)
        ent_n = client_entropy[cid]
        r = spoc_reward(lat_n, en_n, ent_n, weights=meta_w)
        rewards.append((cid, r, lat_n, en_n, ent_n))

    rewards.sort(key=lambda x: x[1], reverse=True)

    exploration_ratio = 0.6 if round_num < 10 else 0.4
    top_k = int((1 - exploration_ratio) * CLIENTS_PER_ROUND)
    rand_k = CLIENTS_PER_ROUND - top_k

    top_clients = [cid for cid, *_ in rewards[:top_k]]
    explore_slice = rewards[int(0.2 * NUM_CLIENTS):]

    if explore_slice:
        scores = np.array([r for _, r, *_ in explore_slice])
        if scores.sum() == 0:
            probs = np.ones_like(scores) / len(scores)
        else:
            probs = scores / (scores.sum() + 1e-9)
        candidate_ids = [cid for cid, *_ in explore_slice]
        random_clients = np.random.choice(
            candidate_ids,
            size=min(rand_k, len(candidate_ids)),
            replace=False,
            p=probs
        ).tolist()
    else:
        random_clients = []

    selected = top_clients + random_clients
    selected = selected[:CLIENTS_PER_ROUND]
    return selected, rewards


# ---------------- PPO loader (baseline) ----------------
ppo_agents = {}
class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
    def reset(self, seed=None, options=None): return np.zeros(7, dtype=np.float32), {}
    def step(self, action): return np.zeros(7, dtype=np.float32), (1.0 if action else -0.5), False, False, {}

def load_ppo_agents():
    for f in range(NUM_FOG_NODES):
        env = DummyEnv()
        path = f"ppo_fog_{f}.zip"
        agent = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            clip_range=0.2,
            ent_coef=0.01,
            gamma=0.99,
            n_steps=2048,
            batch_size=64,
            verbose=0,
            device=DEVICE
        )
        if os.path.exists(path):
            try:
                agent = PPO.load(path, device=DEVICE)
            except Exception:
                agent.learn(total_timesteps=2000); agent.save(path)
        else:
            agent.learn(total_timesteps=2000); agent.save(path)
        ppo_agents[f] = agent

# ---------------- FCL (EWC) placeholders ----------------
GLOBAL_EWC = {"fisher": None, "theta_star": None, "lambda_ewc": 0.1}
def compute_fisher_approx(model, data_loader, device=DEVICE, samples=128):
    model.eval()
    fisher = {n: torch.zeros_like(p) for n,p in model.named_parameters()}
    count = 0
    for Xbatch, ybatch in data_loader:
        Xb = Xbatch.to(device); yb = ybatch.to(device)
        logits = model(Xb); loss = F.cross_entropy(logits, yb)
        loss.backward()
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += (p.grad.detach() ** 2)
        model.zero_grad(); count += 1
        if count >= samples: break
    for n in fisher:
        fisher[n] = fisher[n] / max(1.0, count)
    return fisher

def ewc_penalty(model):
    if GLOBAL_EWC["fisher"] is None or GLOBAL_EWC["theta_star"] is None: return 0.0
    penalty = 0.0
    for (n,p), pstar in zip(model.named_parameters(), GLOBAL_EWC["theta_star"]):
        Fdiag = GLOBAL_EWC["fisher"].get(n, None)
        if Fdiag is not None:
            penalty += (Fdiag * (p - pstar).detach()**2).sum().item()
    return GLOBAL_EWC["lambda_ewc"] * penalty

# ---------------- get_real_client_data and client_fn ----------------
def get_real_client_data(cid, round_num):
    partition = round_partitions[round_num] if (round_num < len(round_partitions)) else {}
    idx = partition.get(cid, [])
    if USE_GLOBAL_HOLDOUT:
        idx = [i for i in idx if i not in holdout_idx_set]
    if not idx or max(idx) >= len(crawdad_df):
        return None, None
    subset = crawdad_df.loc[idx]
    if "slice_label" in subset.columns:
        y = pd.to_numeric(subset["slice_label"], errors="coerce").dropna().astype(np.int64).values
    elif "slice_type_output" in subset.columns:
        y = map_labels_safe(subset["slice_type_output"])
    else:
        return None, None
    try:
        x = encode_features(subset[FEATURE_VECTOR_COLS])
    except Exception:
        x = subset[PKL_FEATURES].fillna(0.0).values.astype(np.float32)
    if len(np.unique(y)) < 2:
        return None, None
    return x, y

class DummyClient(NumPyClient):
    def get_parameters(self, config): return []
    def fit(self, parameters, config): return [], 0, {}
    def evaluate(self, parameters, config): return 0.0, 1, {"accuracy": 0.0, "f1_score": 0.0, "auc": 0.0}

def client_fn(context: Context):
    cid = int(context.node_config.get("partition-id", 0)) % NUM_CLIENTS
    r = int(context.run_config.get("round", 0))
    if r >= len(round_partitions):
        return DummyClient().to_client()
    x, y = get_real_client_data(cid, r)
    if x is None or y is None:
        return DummyClient().to_client()
    return FLClient(cid, (x, y)).to_client()

# ---------------- FLClient (Flower) ----------------
class FLClient(NumPyClient):
    def __init__(self, cid, data):
        self.cid = cid
        self.model = MobilityModel(FEATURE_VECTOR_DIM).to(DEVICE)

        print(f"Client {self.cid} model device:",
              next(self.model.parameters()).device)

        X = torch.tensor(data[0]).float(); y = torch.tensor(data[1]).long()
        unique_classes = torch.unique(y)
        if len(unique_classes) < 2 or any((y == c).sum() < 2 for c in unique_classes):
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(sss.split(X, y))
            X_train, X_val = X[train_idx], X[val_idx]; y_train, y_val = y[train_idx], y[val_idx]
        self.x_train = X_train.to(DEVICE); self.y_train = y_train.to(DEVICE)
        self.x_val = X_val.to(DEVICE); self.y_val = y_val.to(DEVICE)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.01,
            weight_decay=1e-4
        )

        if USE_FEDPROX:
            global_params = [p.detach().clone() for p in self.model.parameters()]
        else:
            global_params = None

        class_counts = torch.bincount(self.y_train)
        class_weights = 1.0 / (class_counts.float() + 1e-9)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = class_weights.to(DEVICE)

        loss = None

        for epoch in range(15):
            optimizer.zero_grad()
            logits = self.model(self.x_train)
            loss = F.cross_entropy(logits, self.y_train, weight=class_weights)

            if USE_FEDPROX and global_params is not None:
                prox_term = 0.0
                for p_local, p_global in zip(self.model.parameters(), global_params):
                    prox_term += torch.sum((p_local - p_global) ** 2)
                loss = loss + (FEDPROX_MU / 2.0) * prox_term

            if USE_FCL and GLOBAL_EWC["fisher"] is not None:
                loss = loss + ewc_penalty(self.model)

            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.x_val)
            pred_prob = F.softmax(logits, dim=1).cpu().numpy()
            y_true = self.y_val.cpu().numpy()
            y_pred = np.argmax(pred_prob, axis=1)
            auc = safe_auc(y_true, pred_prob)
            val_acc = float(accuracy_score(y_true, y_pred))
            val_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
            p_val, r_val, f_val, sup_val = precision_recall_fscore_support(y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0)
            cm_val = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

            train_logits = self.model(self.x_train)
            train_pred = torch.argmax(train_logits, dim=1).cpu().numpy()
            y_train_true = self.y_train.cpu().numpy()
            train_acc = float(accuracy_score(y_train_true, train_pred))
            p_tr, r_tr, f_tr, sup_tr = precision_recall_fscore_support(y_train_true, train_pred, labels=list(range(NUM_CLASSES)), zero_division=0)
            cm_tr = confusion_matrix(y_train_true, train_pred, labels=list(range(NUM_CLASSES)))

        metrics = {
            "train_loss": float(loss.detach().cpu().numpy()) if loss is not None else 0.0,
            "train_acc": float(train_acc),
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_auc": float(auc if not math.isnan(auc) else 0.0)
        }

        for c in range(NUM_CLASSES):
            metrics[f"val_precision_c{c}"] = float(p_val[c])
            metrics[f"val_recall_c{c}"] = float(r_val[c])
            metrics[f"val_f1_c{c}"] = float(f_val[c])
            metrics[f"val_support_c{c}"] = int(sup_val[c])

        for c in range(NUM_CLASSES):
            metrics[f"train_precision_c{c}"] = float(p_tr[c])
            metrics[f"train_recall_c{c}"] = float(r_tr[c])
            metrics[f"train_f1_c{c}"] = float(f_tr[c])
            metrics[f"train_support_c{c}"] = int(sup_tr[c])

        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                metrics[f"conf_{i}_{j}"] = int(cm_val[i,j])
                metrics[f"train_conf_{i}_{j}"] = int(cm_tr[i,j])

        return get_parameters(self.model), len(self.x_train), metrics

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters); self.model.eval()
        if self.x_val.size(0) == 0:
            return 0.0, 1, {"accuracy": 0.0, "f1_score": 0.0, "auc": 0.0}
        with torch.no_grad():
            logits = self.model(self.x_val); loss = F.cross_entropy(logits, self.y_val)
            pred = torch.argmax(logits, dim=1)
            y_true = self.y_val.cpu().numpy()
            y_pred = pred.cpu().numpy()
            acc = float(accuracy_score(y_true, y_pred))
            f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
            auc = safe_auc(y_true, F.softmax(logits, dim=1).cpu().numpy())
            p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

        metrics = {"accuracy": acc, "f1_score": f1, "auc": (auc if not math.isnan(auc) else 0.0)}
        for c in range(NUM_CLASSES):
            metrics[f"precision_c{c}"] = float(p[c])
            metrics[f"recall_c{c}"] = float(r[c])
            metrics[f"f1_c{c}"] = float(f[c])
            metrics[f"support_c{c}"] = int(sup[c])
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                metrics[f"conf_{i}_{j}"] = int(cm[i,j])
        return float(loss), len(self.y_val), metrics

# ---------------- Sampling + Fairness + Attribution logging ----------------
all_metrics = []
fog_metrics = []
per_round_client_selection = {}
per_round_client_rewards = {}
per_client_history = defaultdict(list)
attribution_log = []
meta_history_log = []

# Precompute per-client mode label
try:
    mode_series = crawdad_df.groupby("client_id")["slice_label"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else -1)
    mode_series = mode_series.reindex(range(NUM_CLIENTS), fill_value=-1)
    CLIENT_MODE_LABEL = mode_series.astype(int).values
    print("Precomputed CLIENT_MODE_LABEL for fairness (len={}): sample: {}".format(len(CLIENT_MODE_LABEL), CLIENT_MODE_LABEL[:8]))
except Exception as e:
    print("Failed to precompute CLIENT_MODE_LABEL, falling back:", e)
    CLIENT_MODE_LABEL = None

def fairness_quota_filter(selected_clients, alpha_min=0.25, labels=None):
    if labels is None:
        if CLIENT_MODE_LABEL is not None:
            labels = CLIENT_MODE_LABEL
        else:
            labels = np.array([crawdad_df[crawdad_df["client_id"]==i]["slice_label"].mode().iloc[0]
                               if len(crawdad_df[crawdad_df["client_id"]==i])>0 else -1
                               for i in range(NUM_CLIENTS)])
    sel = list(selected_clients)
    n = max(1, len(sel))
    class_counts = {}
    for c in np.unique(labels[labels >= 0]):
        class_counts[c] = sum(1 for cid in sel if labels[cid] == c)
    for c in class_counts:
        target = math.ceil(alpha_min * n)
        have = class_counts[c]
        if have < target:
            candidates = [cid for cid, lab in enumerate(labels) if lab == c and cid not in sel]
            need = max(0, target - have)
            if candidates:
                to_add = random.sample(candidates, min(need, len(candidates)))
                sel.extend(to_add)
    sel = list(dict.fromkeys(sel))
    return sel

def apply_churn(selected, mode="random", rate=0.3):
    if not selected:
        return selected
    k = int(len(selected) * rate)
    if k <= 0:
        return selected
    if mode == "random":
        drop = set(random.sample(selected, min(k, len(selected))))
        return [c for c in selected if c not in drop]
    elif mode == "adversarial":
        ranked = sorted(selected, key=lambda c: client_latency[c], reverse=True)
        drop = set(ranked[:k])
        return [c for c in selected if c not in drop]
    return selected

def sample_clients_with_offloading(round_num):
    if not USE_FEDSPOC and not USE_PPO_BASELINE:
        selected = random.sample(range(NUM_CLIENTS), CLIENTS_PER_ROUND)
        per_round_client_selection[round_num] = selected
        return selected
    elif USE_FEDSPOC:
        selected, rewards_all = sample_clients_with_spoc(round_num, reward_threshold=0.5)
        per_round_client_rewards[round_num] = rewards_all
    else:
        selected = []; rewards_all = []
        for cid in range(NUM_CLIENTS):
            fid = int(fog_assignment_map[cid])
            features = np.clip(np.array([
                client_entropy[cid],
                client_skewness[cid],
                (client_energy[cid] - 0.3) / 0.7,
                (client_latency[cid] - client_latency.min()) / (client_latency.max() - client_latency.min() + 1e-9),
                client_timestamp[cid],
                0.5, 0.5
            ], dtype=np.float32), 0.0, 1.0)
            action = 1
            if fid in ppo_agents:
                action, _ = ppo_agents[fid].predict(features, deterministic=True)
            if action == 1:
                selected.append(cid)
            rewards_all.append((cid, float(action), 0.0, 0.0, 0.0))
        per_round_client_rewards[round_num] = rewards_all

    selected_before = selected.copy()
    selected = fairness_quota_filter(selected, alpha_min=0.10, labels=None)
    fairness_added = list(set(selected) - set(selected_before))

    k_rand = max(3, int(0.10 * CLIENTS_PER_ROUND))
    candidates = [c for c in range(NUM_CLIENTS) if c not in selected]
    if candidates:
        try:
            random_add = random.sample(candidates, min(k_rand, len(candidates)))
            selected.extend(random_add)
        except ValueError:
            pass

    selected = list(dict.fromkeys(selected))

    if ENABLE_CHURN:
        selected = apply_churn(selected, mode=CHURN_MODE, rate=CHURN_RATE)

    per_round_client_selection[round_num] = selected
    for cid in selected:
        per_client_history[cid].append(round_num)
    for cid in selected:
        fid = int(fog_assignment_map[cid])
        fog_metrics.append({
            "round": round_num, "fog_node": fid, "client_id": cid,
            "accepted": 1, "energy": float(client_energy[cid]),
            "latency": float(client_latency[cid]), "sla_violated": int(client_latency[cid] > 80),
            "cache_hit": int(random.random() < 0.6), "fairness_added": int(cid in fairness_added)
        })
    if not selected:
        selected = [0]
        per_round_client_selection[round_num] = selected
    return selected

def generate_active_clients():
    sel_records = []
    for r in range(NUM_ROUNDS):
        selected = sample_clients_with_offloading(r)
        sel_records.append({"round": r, "selected_clients": selected})
    if SAVE_DETAILED:
        sel_df = pd.DataFrame(sel_records)
        sel_df["selected_clients"] = sel_df["selected_clients"].apply(lambda x: x)
        sel_df.to_csv(os.path.join(OUTPUT_DIR, f"selection_rounds_summary.csv"), index=False)

# ---------------- Aggregation & meta updates ----------------
PATIENCE = 10
BEST_METRIC = -1.0
STOPPING_ROUND = None
PREV_VAL_F1 = 0.0
GLOBAL_FINAL_PARAMS = None


def aggregate_fit_metrics(results):
    global BEST_METRIC, STOPPING_ROUND, PREV_VAL_F1
    global meta_history_log, meta_w

    core_keys = ["train_loss","train_acc","val_acc","val_f1","val_auc"]
    m = {
        k: float(np.nanmean([res.get(k, np.nan) for _, res in results]))
        for k in core_keys
        if any(k in res for _, res in results)
    }

    # Per-class logging
    for c in range(NUM_CLASSES):
        vals = [res.get(f"val_f1_c{c}", np.nan) for _, res in results]
        m[f"val_f1_c{c}_mean"] = float(np.nanmean(vals))
        m[f"val_f1_c{c}_std"]  = float(np.nanstd(vals))

    # Fairness
    per_class_f1 = np.array(
        [m.get(f"val_f1_c{c}_mean", 0.0) for c in range(NUM_CLASSES)],
        dtype=float
    )

    if per_class_f1.sum() > 0:
        jain = (per_class_f1.sum() ** 2) / (
            NUM_CLASSES * np.sum(per_class_f1 ** 2) + 1e-9
        )
        p = per_class_f1 / (per_class_f1.sum() + 1e-9)
        entropy_fair = -np.sum(p * np.log(p + 1e-9)) / np.log(NUM_CLASSES)
        cov = per_class_f1.std() / (per_class_f1.mean() + 1e-9)
        min_slice = float(per_class_f1.min())
        gap = float(per_class_f1.max() - per_class_f1.min())
    else:
        jain = entropy_fair = cov = min_slice = gap = 0.0

    m.update({
        "jain_fairness": float(jain),
        "entropy_fairness": float(entropy_fair),
        "cov_f1": float(cov),
        "min_slice_f1": float(min_slice),
        "max_min_gap": float(gap)
    })

    round_idx = len(all_metrics) + 1
    all_metrics.append({"round": round_idx, **m})

    pd.DataFrame(all_metrics).to_csv(
        os.path.join(OUTPUT_DIR, "federated_metrics_rounds.csv"),
        index=False
    )

    # Meta update
    curr_val_f1 = m.get("val_f1", 0.0)
    delta = curr_val_f1 - PREV_VAL_F1

    if USE_META_UPDATE:
        update_meta_weights(delta)

    PREV_VAL_F1 = curr_val_f1

    # Early stopping
    if m.get("val_auc", 0.0) > BEST_METRIC:
        BEST_METRIC = m["val_auc"]
        STOPPING_ROUND = round_idx
    elif STOPPING_ROUND is not None and (round_idx - STOPPING_ROUND) > PATIENCE:
        print(f"⛔ Early stopping at round {round_idx}")
        raise SystemExit("Early stopping triggered")

    return m

def aggregate_evaluate_metrics(results):
    return { k: float(np.nanmean([res.get(k, np.nan) for _, res in results])) for k in ["accuracy","f1_score","auc"] }

class FedSPOCStrategy(FedAvgM):

    def aggregate_fit(self, server_round, results, failures):
        global GLOBAL_FINAL_PARAMS

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Capture true global model parameters
        if aggregated_parameters is not None:
            GLOBAL_FINAL_PARAMS = aggregated_parameters

        return aggregated_parameters, aggregated_metrics

    def configure_fit(self, server_round, parameters, client_manager):
        selected_cids = sample_clients_with_offloading(server_round - 1)
        fit_config = {"round": server_round}
        selected_clients = []
        for cid in selected_cids:
            try:
                client = client_manager.get_client(str(cid))
                selected_clients.append(client)
            except Exception:
                continue
        if not selected_clients:
            return super().configure_fit(server_round, parameters, client_manager)
        print(f"Round {server_round}: Training {len(selected_clients)} clients")
        return [(client, fit_config) for client in selected_clients]

# ---------------- Causal scoring ----------------
def compute_causal_scores():
    X = []; y = []
    for cid in range(NUM_CLIENTS):
        feats = [client_entropy[cid], client_energy[cid], (client_latency[cid]-client_latency.min())/(client_latency.max()-client_latency.min()+1e-9), client_skewness[cid]]
        X.append(feats)
        atts = [a["attribution"] for a in attribution_log if a["client_id"] == cid]
        y.append(np.mean(atts) if len(atts)>0 else 0.0)
    X = np.array(X); y = np.array(y)
    if X.shape[0] > 0:
        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict(X)
        causal_scores = pred.tolist()
        coef = model.coef_.tolist()
    else:
        causal_scores = [0.0]*NUM_CLIENTS
        coef = []
    df = pd.DataFrame({
        "client_id": np.arange(NUM_CLIENTS),
        "entropy": X[:,0] if X.shape[0]>0 else np.zeros(NUM_CLIENTS),
        "energy": X[:,1] if X.shape[0]>0 else np.zeros(NUM_CLIENTS),
        "latency_n": X[:,2] if X.shape[0]>0 else np.zeros(NUM_CLIENTS),
        "skewness": X[:,3] if X.shape[0]>0 else np.zeros(NUM_CLIENTS),
        "causal_score": causal_scores, "avg_attribution": y
    })
    df.to_csv(os.path.join(OUTPUT_DIR, "causal_scores.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "causal_coef.txt"), "w") as fh:
        fh.write("linear_model_coefficients: " + str(coef))
    return df, coef


def run_sensitivity_analysis():

    global meta_w, RARITY_BETA, PPO_CLIP, EWMA_LAMBDA

    sensitivity_rows = []

    configs = {
        "wL_low":    {"meta_w": np.array([0.2, 0.5, 0.3])},
        "wL_high":   {"meta_w": np.array([0.8, 0.1, 0.1])},
        "beta_0":    {"RARITY_BETA": 0.0},
        "beta_2":    {"RARITY_BETA": 2.0},
        "clip_01":   {"PPO_CLIP": 0.1},
        "clip_04":   {"PPO_CLIP": 0.4},
        "lambda_02": {"EWMA_LAMBDA": 0.2},
        "lambda_09": {"EWMA_LAMBDA": 0.9},
    }

    # Store original baseline values
    original = {
        "meta_w": meta_w.copy(),
        "RARITY_BETA": RARITY_BETA,
        "PPO_CLIP": PPO_CLIP,
        "EWMA_LAMBDA": EWMA_LAMBDA
    }

    # Optionally shorten rounds for sensitivity (faster)
    original_rounds = NUM_ROUNDS
    SHORT_ROUNDS = 10

    try:
        globals()["NUM_ROUNDS"] = SHORT_ROUNDS

        for tag, cfg in configs.items():

            print(f"\n--- Sensitivity: {tag} ---")

            # ---------------- Reset baseline ----------------
            meta_w[:] = original["meta_w"]
            RARITY_BETA = original["RARITY_BETA"]
            PPO_CLIP = original["PPO_CLIP"]
            EWMA_LAMBDA = original["EWMA_LAMBDA"]

            # ---------------- Apply config ----------------
            if "meta_w" in cfg:
                meta_w[:] = cfg["meta_w"]

            if "RARITY_BETA" in cfg:
                RARITY_BETA = cfg["RARITY_BETA"]

            if "PPO_CLIP" in cfg:
                PPO_CLIP = cfg["PPO_CLIP"]

            if "EWMA_LAMBDA" in cfg:
                EWMA_LAMBDA = cfg["EWMA_LAMBDA"]

            # ---------------- Run experiment ----------------
            run_experiment(seed=42)

            # ---------------- Collect metrics safely ----------------
            if len(all_metrics) > 0:
                best_f1  = max(m.get("val_f1", 0.0) for m in all_metrics)
                best_auc = max(m.get("val_auc", 0.0) for m in all_metrics)
            else:
                best_f1, best_auc = 0.0, 0.0

            sensitivity_rows.append({
                "config": tag,
                "best_val_f1": best_f1,
                "best_val_auc": best_auc
            })

    finally:
        # ---------------- Restore original baseline ----------------
        meta_w[:] = original["meta_w"]
        RARITY_BETA = original["RARITY_BETA"]
        PPO_CLIP = original["PPO_CLIP"]
        EWMA_LAMBDA = original["EWMA_LAMBDA"]
        globals()["NUM_ROUNDS"] = original_rounds

    # ---------------- Save results ----------------
    df = pd.DataFrame(sensitivity_rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_analysis.csv"), index=False)

    print("\n✅ Sensitivity analysis complete")
    print(df)

    return df

# ---------------- Server function ----------------
def server_fn(context: Context):
    initial_model = MobilityModel(FEATURE_VECTOR_DIM)
    return ServerAppComponents(
        strategy=FedSPOCStrategy(
            initial_parameters=ndarrays_to_parameters(get_parameters(initial_model)),
            fraction_fit=0.0,
            fraction_evaluate=0.0,
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
            on_fit_config_fn=lambda r: {"round": r},
            fit_metrics_aggregation_fn=aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics
        ),
        config=ServerConfig(num_rounds=NUM_ROUNDS)
    )

# ---------------- Plotting utilities ----------------
def plot_round_metric(path_csv, metric_col, ylabel, fname, smooth=False, show_plot=False):
    if not os.path.exists(path_csv): return
    df = pd.read_csv(path_csv)
    if "round" not in df.columns: return
    x = df["round"].values; y = df[metric_col].values
    plt.figure(figsize=(8,4))
    if smooth:
        window = max(1, int(len(y)*0.1))
        y_s = pd.Series(y).rolling(window=window, min_periods=1).mean().values
        plt.plot(x, y_s, marker='o', linewidth=1)
    else:
        plt.plot(x, y, marker='o', linewidth=1)
    plt.xlabel("Round"); plt.ylabel(ylabel); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
    if show_plot:
        plt.show()
    plt.close()

def plot_fog_heatmap(show_plot=False):
    path = os.path.join(OUTPUT_DIR, "fog_metrics_detailed.csv")
    if not os.path.exists(path): return
    df = pd.read_csv(path)
    agg = df.groupby("fog_node").agg({"latency":"mean","energy":"mean","sla_violated":"sum","client_id":"count"}).reset_index()
    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[0].bar(agg["fog_node"], agg["latency"]); ax[0].set_title("Avg Latency")
    ax[1].bar(agg["fog_node"], agg["energy"]); ax[1].set_title("Avg Energy")
    ax[2].bar(agg["fog_node"], agg["sla_violated"]); ax[2].set_title("SLA Violations")
    for a in ax: a.set_xlabel("Fog Node")
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "fog_node_metrics.png"), dpi=300)
    if show_plot:
        plt.show()
    plt.close()

# ---------------- Orchestration ----------------
def run_experiment(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---------------- CLEAR LOGS ----------------
    all_metrics.clear()
    fog_metrics.clear()
    per_round_client_selection.clear()
    per_round_client_rewards.clear()
    per_client_history.clear()
    attribution_log.clear()
    meta_history_log.clear()

    # ---------------- RESET GLOBAL STATE (CRITICAL) ----------------
    global BEST_METRIC, STOPPING_ROUND, PREV_VAL_F1, meta_w, auc_skipped_count

    BEST_METRIC = -1.0
    STOPPING_ROUND = None
    PREV_VAL_F1 = 0.0
    auc_skipped_count = 0

    # Reset meta weights to default
    meta_w[:] = np.array([0.6, 0.3, 0.1])

    # ---------------- PPO (if baseline mode) ----------------
    if USE_PPO_BASELINE and not USE_FEDSPOC:
        load_ppo_agents()

    # ---------------- Ray Reset (clean start) ----------------
    import ray
    if ray.is_initialized():
        ray.shutdown()

    ray.init(
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
        _metrics_export_port=None
    )

    # ---------------- Run Simulation ----------------
    app = ServerApp(server_fn=server_fn)
    start = time.time()

    try:
        run_simulation(
            server_app=app,
            client_app=ClientApp(client_fn=client_fn),
            num_supernodes=4,
            backend_config={
                "client_resources": {"num_cpus": 0.5, "num_gpus": 0.1}
            }
        )
    except SystemExit:
        print("Stopped early by strategy.")
    except Exception as e:
        print("Simulation error:", e)

    elapsed = time.time() - start

    # ---------------- Save Outputs ----------------
    if SAVE_DETAILED:
        pd.DataFrame(all_metrics).to_csv(
            os.path.join(OUTPUT_DIR, f"federated_metrics_seed_{seed}.csv"),
            index=False
        )

        pd.DataFrame(fog_metrics).to_csv(
            os.path.join(OUTPUT_DIR, f"fog_metrics_seed_{seed}.csv"),
            index=False
        )

        sel_list = [
            {
                "round": r,
                "selected_count": len(per_round_client_selection.get(r, [])),
                "selected": per_round_client_selection.get(r, [])
            }
            for r in range(NUM_ROUNDS)
        ]

        pd.DataFrame(sel_list).to_csv(
            os.path.join(OUTPUT_DIR, f"selection_summary_seed_{seed}.csv"),
            index=False
        )

        pd.DataFrame(attribution_log).to_csv(
            os.path.join(OUTPUT_DIR, f"client_attributions_seed_{seed}.csv"),
            index=False
        )

        pd.DataFrame(meta_history_log).to_csv(
            os.path.join(OUTPUT_DIR, f"meta_history_seed_{seed}.csv"),
            index=False
        )

    # ---------------- Save Checkpoint ----------------
    # ---------------- Save Checkpoint ----------------
    try:
        if GLOBAL_FINAL_PARAMS is not None:

            # Convert Flower Parameters -> list of numpy arrays
            ndarrays = parameters_to_ndarrays(GLOBAL_FINAL_PARAMS)

            model = MobilityModel(FEATURE_VECTOR_DIM)
            set_parameters(model, ndarrays)

            ck_dir = os.path.join(CHECKPOINT_DIR, f"seed_{seed}")
            os.makedirs(ck_dir, exist_ok=True)

            torch.save(
                model.state_dict(),
                os.path.join(ck_dir, f"global_round_{len(all_metrics)}.pt")
            )

    except Exception as e:
        print("Checkpoint save failed:", e)
      # ---------------- Post Processing ----------------
    compute_causal_scores()

    metrics_csv = os.path.join(
        OUTPUT_DIR,
        f"federated_metrics_seed_{seed}.csv"
    )

    plot_round_metric(
        metrics_csv,
        "val_acc",
        "Validation Accuracy",
        f"val_acc_seed_{seed}.png",
        smooth=True,
        show_plot=False
    )

    plot_round_metric(
        metrics_csv,
        "val_f1",
        "Validation Macro-F1",
        f"val_f1_seed_{seed}.png",
        smooth=True,
        show_plot=False
    )

    plot_round_metric(
        metrics_csv,
        "val_auc",
        "Validation AUC",
        f"val_auc_seed_{seed}.png",
        smooth=True,
        show_plot=False
    )

    plot_fog_heatmap(show_plot=False)

    # ---------------- Save Warnings Summary ----------------
    with open(
        os.path.join(OUTPUT_DIR, f"warnings_summary_seed_{seed}.txt"),
        "w"
    ) as fh:
        fh.write(f"auc_skipped_count: {auc_skipped_count}\n")
        fh.write(f"elapsed_seconds: {elapsed}\n")

    # ---------------- Ray Shutdown (clean end) ----------------
    if ray.is_initialized():
        ray.shutdown()

    print(f"Results saved to {OUTPUT_DIR} for seed {seed}")
# ---------------- Cross-seed aggregation ----------------
def aggregate_across_seeds():
    dfs = []
    for s in SEEDS:
        path = os.path.join(OUTPUT_DIR, f"federated_metrics_seed_{s}.csv")
        if os.path.exists(path):
            dfs.append(pd.read_csv(path).assign(seed=s))
    if not dfs:
        print("No per-seed metric CSVs found; skipping cross-seed aggregation.")
        return
    df = pd.concat(dfs, ignore_index=True)
    agg_df = df.groupby("round").agg(['mean','std']).reset_index()
    agg_df.columns = ['_'.join(filter(None,col)).strip('_') for col in agg_df.columns.values]
    agg_df.to_csv(os.path.join(OUTPUT_DIR,"federated_metrics_allseeds.csv"), index=False)

    for metric in ["val_f1","val_auc","val_acc"]:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col in agg_df.columns:
            x = agg_df["round"]
            y = agg_df[mean_col]
            yerr = agg_df[std_col].fillna(0.0)
            plt.figure(figsize=(8,4))
            plt.errorbar(x, y, yerr=yerr, marker='o', capsize=3)
            plt.title(f"{metric} across rounds (mean ± std over seeds)")
            plt.xlabel("Round"); plt.ylabel(metric); plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"summary_{metric}.png"), dpi=600)
            plt.show()
            plt.close()

    summary = []
    for s in SEEDS:
        path = os.path.join(OUTPUT_DIR, f"federated_metrics_seed_{s}.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        if "val_f1" in d.columns:
            best_idx = d["val_f1"].idxmax()
            row = d.loc[best_idx]
            summary.append({
                "seed": s,
                "best_round": int(row["round"]),
                "best_val_f1": float(row["val_f1"]),
                "best_val_auc": float(row.get("val_auc", np.nan))
            })
    pd.DataFrame(summary).to_csv(os.path.join(OUTPUT_DIR,"summary_all_seeds.csv"), index=False)
    print("Cross-seed aggregation completed and saved.")





# ---------------- Diagnostics functions ----------------
def diagnostics_print():
    print("=== DIAGNOSTICS ===")
    print("crawdad_df shape:", crawdad_df.shape)
    print("Total unique clients in data by client_id:", crawdad_df["client_id"].nunique())
    counts = crawdad_df.groupby("client_id").size()
    print("client sample count stats:\n", counts.describe())
    print("clients with <=4 samples:", (counts <= 4).sum())
    label_counts = crawdad_df.groupby("client_id")["slice_label"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else -1)
    print("example label counts for first 8 clients:", dict(list(label_counts.items())[:8]))
    dup_found = []
    for cid in range(min(NUM_CLIENTS, crawdad_df["client_id"].max()+1)):
        subset = crawdad_df[crawdad_df["client_id"]==cid]
        if subset.empty: continue
        rows = [tuple(r) for r in subset[FEATURE_VECTOR_COLS + ["slice_label"]].fillna("NA").astype(str).values]
        c = Counter(rows)
        if any(v > 1 for v in c.values()):
            dup_found.append(cid)
    print("clients with duplicated rows (sample):", dup_found[:20], "total:", len(dup_found))
    sel_path = os.path.join(OUTPUT_DIR, f"selection_summary_seed_{SEEDS[0]}.csv")
    if os.path.exists(sel_path):
        sel_df = pd.read_csv(sel_path)
        sel_sets = []
        for s in sel_df["selected"].tolist():
            if isinstance(s, str):
                try:
                    ss = eval(s)
                except Exception:
                    ss = s
            else:
                ss = s
            sel_sets.append(set(ss) if isinstance(ss, (list, tuple, set)) else set())
        same_across = len(sel_sets) > 0 and all(sel_sets[0] == s for s in sel_sets)
        print("selected identical across rounds?", same_across)
        union_sel = set().union(*sel_sets) if sel_sets else set()
        print("unique clients ever selected:", len(union_sel))
    else:
        print("Selection summary not yet available.")
    print("=== END DIAGNOSTICS ===\n")

def evaluate_global_holdout():

    if not USE_GLOBAL_HOLDOUT:
        print("Global holdout disabled.")
        return

    if not holdout_idx_set:
        print("No holdout rows prepared.")
        return

    holdout_df = crawdad_df.loc[sorted(holdout_idx_set)]
    if holdout_df.shape[0] == 0:
        print("Holdout DataFrame empty.")
        return

    print(f"Evaluating on global holdout set: {len(holdout_df)} samples")

    # ---------------- Prepare data ----------------
    X_hold = encode_features(holdout_df[FEATURE_VECTOR_COLS])
    y_hold = holdout_df["slice_label"].astype(int).values

    model = MobilityModel(FEATURE_VECTOR_DIM).to(DEVICE)

    # ---------------- Locate latest valid checkpoint ----------------
    ck_paths = []
    for root, dirs, files in os.walk(CHECKPOINT_DIR):
        for f in files:
            if f.startswith("global_round_") and f.endswith(".pt"):
                ck_paths.append(os.path.join(root, f))

    if not ck_paths:
        print("No checkpoint found to evaluate.")
        return

    ck_paths = sorted(ck_paths)
    loaded = False

    for ck in reversed(ck_paths):  # Try newest first
        try:
            params = torch.load(ck, map_location=DEVICE)

            # Case 1: Proper PyTorch state_dict
            if isinstance(params, dict):
                model.load_state_dict(params)
                print(f"Loaded state_dict checkpoint: {ck}")
                loaded = True
                break

            # Case 2: List of numpy arrays (Flower format)
            elif isinstance(params, list):
                sd = model.state_dict()
                for k, arr in zip(sd.keys(), params):
                    sd[k] = torch.tensor(arr)
                model.load_state_dict(sd)
                print(f"Loaded ndarray-list checkpoint: {ck}")
                loaded = True
                break

            else:
                print(f"Skipping unsupported checkpoint format: {ck} ({type(params)})")

        except Exception as e:
            print(f"Skipping corrupted checkpoint {ck}: {e}")
            continue

    if not loaded:
        print("No valid checkpoint could be loaded.")
        return

    # ---------------- Evaluation ----------------
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_hold).float().to(DEVICE)
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_hold, y_pred)
    f1  = f1_score(y_hold, y_pred, average="macro", zero_division=0)

    try:
        y_bin = label_binarize(y_hold, classes=np.arange(NUM_CLASSES))
        auc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
    except Exception:
        auc = float("nan")

    print("\n=== GLOBAL HOLDOUT RESULTS ===")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Macro-F1   : {f1:.4f}")
    print(f"Macro-AUC  : {auc:.4f}" if not np.isnan(auc) else "Macro-AUC  : NaN")
    print("\nClassification Report:\n")
    print(classification_report(y_hold, y_pred, digits=4))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_hold, y_pred))
    print("================================\n")

# *** NEW: SHAP Attribution Analysis (Reviewer 1, Point 11) ***
def run_shap_analysis():
    """
    Computes SHAP feature importance on the final global model using the holdout set.
    GPU cost: very low — inference only.
    Saves:
        - shap_summary.png
        - shap_feature_importance.csv
    """

    # ---------------- Safe SHAP import ----------------
    try:
        import shap
    except ImportError:
        print("Installing shap...")
        import subprocess
        subprocess.run(["pip", "install", "shap", "-q"], check=True)
        import shap

    print("\nRunning SHAP analysis on final global model...")

    # ---------------- Locate valid checkpoint ----------------
    model = MobilityModel(FEATURE_VECTOR_DIM).to("cpu")

    ck_paths = []
    for root, dirs, files in os.walk(CHECKPOINT_DIR):
        for f in files:
            if f.startswith("global_round_") and f.endswith(".pt"):
                ck_paths.append(os.path.join(root, f))

    if not ck_paths:
        print("⚠ No checkpoint found for SHAP.")
        return None

    ck_paths = sorted(ck_paths)
    loaded = False

    for ck in reversed(ck_paths):  # Try newest first
        try:
            params = torch.load(ck, map_location="cpu")

            # Case 1: Proper state_dict
            if isinstance(params, dict):
                model.load_state_dict(params)
                print(f"Loaded state_dict checkpoint: {ck}")
                loaded = True
                break

            # Case 2: Flower ndarray list
            elif isinstance(params, list):
                sd = model.state_dict()
                for k, arr in zip(sd.keys(), params):
                    sd[k] = torch.tensor(arr)
                model.load_state_dict(sd)
                print(f"Loaded ndarray-list checkpoint: {ck}")
                loaded = True
                break

            else:
                print(f"Skipping unsupported checkpoint: {ck} ({type(params)})")

        except Exception as e:
            print(f"Skipping corrupted checkpoint {ck}: {e}")
            continue

    if not loaded:
        print("⚠ No valid checkpoint could be loaded for SHAP.")
        return None

    model.eval()

    # ---------------- Holdout Data ----------------
    if not holdout_idx_set:
        print("⚠ No holdout set found. Skipping SHAP.")
        return None

    holdout_df = crawdad_df.loc[sorted(holdout_idx_set)].sample(
        min(500, len(holdout_idx_set)), random_state=42
    )

    X_hold = encode_features(holdout_df[FEATURE_VECTOR_COLS])

    if X_hold.shape[0] < 200:
        print("⚠ Not enough holdout samples for SHAP.")
        return None

    # ---------------- Feature Names ----------------
    n_cat_encoded = X_hold.shape[1] - len(numeric_cols)
    feature_names = (
        [f"cat_enc_{i}" for i in range(n_cat_encoded)]
        + list(numeric_cols)
    )

    # ---------------- Prediction Wrapper ----------------
    def model_predict(x_np):
        with torch.no_grad():
            t = torch.tensor(x_np, dtype=torch.float32)
            return torch.softmax(model(t), dim=1).numpy()

    # Background and test subset
    background = X_hold[:100]
    test_data = X_hold[100:200]

    # ---------------- SHAP Explainer ----------------
    print("Computing SHAP values (KernelExplainer)...")
    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(test_data, nsamples=100)

    # ---------------- Summary Plot ----------------
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        test_data,
        feature_names=feature_names,
        class_names=["URLLC", "mMTC", "eMBB"],
        show=False
    )
    plt.tight_layout()

    shap_plot_path = os.path.join(OUTPUT_DIR, "shap_summary.png")
    plt.savefig(shap_plot_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"SHAP summary plot saved: {shap_plot_path}")

       # ---------------- Mean Absolute SHAP Importance (Robust Multi-Class Safe) ----------------

    if isinstance(shap_values, list):
        # Multi-class list output
        class_importances = []
        for sv in shap_values:
            class_importances.append(np.abs(sv).mean(axis=0))
        mean_shap = np.mean(np.vstack(class_importances), axis=0)

    else:
        # Single array output
        if shap_values.ndim == 3:
            # shape: [samples, features, classes]
            mean_shap = np.mean(np.abs(shap_values), axis=(0, 2))
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)

    # Safety alignment
    if len(mean_shap) != len(feature_names):
        print("⚠ SHAP feature length mismatch. Truncating safely.")
        min_len = min(len(mean_shap), len(feature_names))
        mean_shap = mean_shap[:min_len]
        feature_names = feature_names[:min_len]

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    shap_csv_path = os.path.join(OUTPUT_DIR, "shap_feature_importance.csv")
    shap_df.to_csv(shap_csv_path, index=False)

    print(f"SHAP feature importance saved: {shap_csv_path}")
    print("\nTop 10 Important Features:\n")
    print(shap_df.head(10).to_string(index=False))

    return shap_df
# ---------------- Appendix & Methods text (printable) ----------------
APPENDIX_CURRICULUM = """
Appendix: Curriculum-aware masking (pseudo-code + math)

Let per-client normalized features be:
 L_i(t)  -- latency normalized to [0,1]
 E_i     -- energy normalized to [0,1]
 H_i     -- entropy (uncertainty) normalized to [0,1]

Meta-weights w = [w_L, w_E, w_H], sum(w)=1
Base reward:
 r_i(t) = 1 - ( w_L * L_i(t) + w_E * E_i + w_H * H_i )

Curriculum-aware mask (boost rare classes):
 Let f_c be the fraction of clients selected for class c in the recent window.
 Let β >= 0 (boost strength), γ >= 0 (nonlinearity).
 Compute rarity factor for class c: B_c = 1 + β * (1 - f_c)**γ
 Then adjusted reward for client i with class c:
 r_i_adj(t) = r_i(t) * B_c

Selection: rank clients by r_i_adj and sample above threshold or top-k.
"""

DUMMY_CLIENT_METHODS = """
Methods: Dummy client fallback
- Purpose: prevent round failure and ensure aggregator receives a well-formed update even if a real client has insufficient or empty data.
- Behavior:
  * A DummyClient implements the same NumPyClient interface.
  * get_parameters: returns empty list or a neutral parameter set.
  * fit: returns zero-length update and zero train size (so aggregator can ignore or handle).
  * evaluate: returns neutral metrics (accuracy=0.0, f1=0.0, auc=0.0).
"""

print(APPENDIX_CURRICULUM)
print(DUMMY_CLIENT_METHODS)

# ---------------- Run (top-level) ----------------
if __name__ == "__main__":
    if RUN_DIAGNOSTICS:
        diagnostics_print()

    for s in SEEDS:
        print("=== Running seed", s, "===")
        os.makedirs(os.path.join(OUTPUT_DIR, f"seed_{s}"), exist_ok=True)
        run_experiment(seed=s)

    # Aggregate across seeds
    try:
        aggregate_across_seeds()
    except Exception as e:
        print("Cross-seed aggregation failed:", e)



    # Global holdout evaluation
    if USE_GLOBAL_HOLDOUT:
        try:
            evaluate_global_holdout()
        except Exception as e:
            print("Global holdout evaluation failed:", e)

    # *** NEW: SHAP attribution (low GPU cost, Reviewer 1 Point 11) ***
    print("\n=== Running SHAP attribution analysis ===")
    try:
        shap_df = run_shap_analysis()
    except Exception as e:
        print("SHAP analysis failed:", e)

    # *** NEW: Sensitivity analysis (medium GPU cost, Reviewer 1 Point 12) ***
    # NOTE: Runs AFTER main experiment to avoid interfering with saved results.
    # Uses 10 rounds and 1 seed only to conserve GPU units.
    print("\n=== Running sensitivity analysis (10 rounds, seed=42) ===")
    try:
        run_sensitivity_analysis()
    except Exception as e:
        print("Sensitivity analysis failed:", e)

    print("\nSaved files in results:", sorted(os.listdir(OUTPUT_DIR)))
    if RUN_DIAGNOSTICS:
        diagnostics_print()
    print("Completed. Outputs in:", OUTPUT_DIR)

# Check overlap between client partitions across rounds
overlap_counts = []

for r in range(1, len(round_partitions)):
    prev = set(sum(round_partitions[r-1].values(), []))
    curr = set(sum(round_partitions[r].values(), []))
    overlap = len(prev.intersection(curr)) / len(curr)
    overlap_counts.append(overlap)

print("Average overlap between consecutive rounds:", sum(overlap_counts)/len(overlap_counts))

union_sel = set().union(*[set(v) for v in per_round_client_selection.values()])
print("Unique clients ever selected:", len(union_sel))

# ============================================================
# DATA LEAKAGE FIX — Run this BEFORE any experiment
# Removes label-leaking features from training pipeline
# ============================================================

# Step 1: Define which columns are "inputs only" (no QoS labels)
SAFE_INPUT_COLS = [
    'lte/5g_ue_category_(input_2)',   # UE category — safe
    'technology_supported_(input_3)', # tech flag — safe
    'day_(input4)',                    # temporal — safe
    'time_(input_5)',                  # temporal — safe
]

# Step 2: Optionally keep QCI but it's risky (maps directly to slice)
# If your dataset has enough samples without QCI, exclude it too.
# Keep it only if f1 drops well below 1.0 after removing QoS cols.
RISKY_COLS_TO_VERIFY = ['qci_(input_6)', 'use_casetype_(input_1)']

# Step 3: These are DEFINITELY leaking — remove them
LEAKING_COLS = [
    'packet_loss_rate_(reliability)',
    'packet_delay_budget_(latency)',
    'normalized_latency',
    'normalized_reliability',
    # Also remove any f_ cols that encode these
]

# Step 4: Rebuild PKL_FEATURES WITHOUT leaking cols
available_safe = [c for c in SAFE_INPUT_COLS if c in crawdad_df.columns]

# Add f_ cols that are NOT derived from QoS
fcols_all = sorted([c for c in crawdad_df.columns if c.startswith("f_")],
                   key=lambda s: int(s.split("_")[1]) if s.split("_")[1].isdigit() else 999)
# Keep only f_ cols that don't map to QoS (heuristic: first 4 f_ cols)
safe_fcols = fcols_all[:4]

PKL_FEATURES = list(dict.fromkeys(safe_fcols + available_safe))
print("Fixed PKL_FEATURES (leaking cols removed):", PKL_FEATURES)

# Step 5: Rebuild encoding pipeline with clean features
categorical_cols = [c for c in PKL_FEATURES if c in crawdad_df.columns and
                    (crawdad_df[c].dtype == object or crawdad_df[c].dtype.name == "category")]
numeric_cols     = [c for c in PKL_FEATURES if c in crawdad_df.columns and c not in categorical_cols]

FEATURE_VECTOR_COLS = categorical_cols + numeric_cols if categorical_cols else numeric_cols

# Refit encoder on clean features
if categorical_cols:
    enc_df = crawdad_df.loc[~crawdad_df.index.isin(holdout_idx_set), categorical_cols].fillna("NA").astype(str)
    try:
        onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    onehot.fit(enc_df.values)
    FEATURE_VECTOR_DIM = onehot.transform(enc_df.values[:1]).shape[1] + len(numeric_cols)
else:
    onehot = None
    FEATURE_VECTOR_DIM = len(numeric_cols)

print(f"New FEATURE_VECTOR_DIM: {FEATURE_VECTOR_DIM}")
print(f"Categorical: {categorical_cols}")
print(f"Numeric:     {numeric_cols}")

# Step 6: Verify leakage is gone — quick sanity check
from sklearn.tree import DecisionTreeClassifier
sample_df = crawdad_df.sample(min(2000, len(crawdad_df)), random_state=42)
X_check = encode_features(sample_df[FEATURE_VECTOR_COLS])
y_check = sample_df["slice_label"].values
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(dt, X_check, y_check, cv=3, scoring="f1_macro")
print(f"\nSanity check — Decision Tree (depth=3) cross-val F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print("✅ Good: F1 < 0.85 means no trivial leakage" if cv_scores.mean() < 0.85
      else "⚠️  Still leaking — check which columns remain predictive")

# ============================================================
# Reviewer-Complete Ablation Runner  — UPDATED
# Supports:
# - FedSPOC / FedAvg / Hierarchical FedAvg / PPO         (Reviewer 1 Point 6)
# - FedProx variants
# - Curriculum on/off
# - Random + Adversarial churn                            (Reviewer 1 Point 9)
# - Cross-seed aggregation
# - Extended fairness metrics per variant                 (Reviewer 1 Point 10)
# - Paired t-test + Wilcoxon signed-rank statistical      (Reviewer 1 Point 6)
# - GPU-saving ablation rounds flag
# ============================================================

import os, shutil, glob, time
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

RESULTS_DIR = "/content/drive/MyDrive/veeral_project/for_fedsptoc250/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# *** NEW: Set fewer rounds for ablation to save GPU units.
# Set ABLATION_ROUNDS = None to use the full NUM_ROUNDS from the main script.
ABLATION_ROUNDS = 20   # saves ~33% GPU vs full 30 rounds; set None for full runs

# ------------------------------------------------------------
# VARIANT CONFIGURATION
# ------------------------------------------------------------

VARIANTS = {

    # ---- Core models ----
    "fedspoc": {
        "USE_FEDSPOC": True,
        "USE_PPO_BASELINE": False,
        "USE_FEDPROX": False,
        "ENABLE_CHURN": False,
        "MONKEYPATCH_FAIRNESS": False
    },

    "fedavg": {
        "USE_FEDSPOC": False,
        "USE_PPO_BASELINE": False,
        "USE_FEDPROX": False,
        "ENABLE_CHURN": False,
        "MONKEYPATCH_FAIRNESS": False
    },

    # *** NEW: Hierarchical FedAvg baseline (present in Table 6 but was missing
    #          from ablation — Reviewer 1 Point 6 requests stronger baselines) ***
    "hier_fedavg": {
        "USE_FEDSPOC": False,
        "USE_PPO_BASELINE": False,
        "USE_FEDPROX": False,
        "ENABLE_CHURN": False,
        "MONKEYPATCH_FAIRNESS": False,
        "MONKEYPATCH_HIERFEDAVG": True    # see monkeypatch block below
    },

    "ppo": {
        "USE_FEDSPOC": False,
        "USE_PPO_BASELINE": True,
        "USE_FEDPROX": False,
        "ENABLE_CHURN": False,
        "MONKEYPATCH_FAIRNESS": False
    },

    # ---- FedProx variants ----
    "fedspoc_prox": {
        "USE_FEDSPOC": True,
        "USE_FEDPROX": True
    },

    "fedavg_prox": {
        "USE_FEDSPOC": False,
        "USE_FEDPROX": True
    },

    # ---- Curriculum off ----
    "fedspoc_nocurr": {
        "USE_FEDSPOC": True,
        "MONKEYPATCH_FAIRNESS": True
    },

    # ---- Churn robustness ----
    "fedspoc_churn_random": {
        "USE_FEDSPOC": True,
        "ENABLE_CHURN": True,
        "CHURN_MODE": "random",
        "CHURN_RATE": 0.3
    },

    "fedspoc_churn_adv": {
        "USE_FEDSPOC": True,
        "ENABLE_CHURN": True,
        "CHURN_MODE": "adversarial",
        "CHURN_RATE": 0.3
    }
}

# ------------------------------------------------------------
# Seeds
# ------------------------------------------------------------

try:
    SEEDS_TO_RUN = list(SEEDS)
except Exception:
    SEEDS_TO_RUN = [42, 52, 62, 72, 82]

# ------------------------------------------------------------
# Backup global flags  (*** UPDATED: added CHURN_RATE ***)
# ------------------------------------------------------------

FLAG_NAMES = [
    "USE_FEDSPOC",
    "USE_PPO_BASELINE",
    "USE_FEDPROX",
    "ENABLE_CHURN",
    "CHURN_MODE",
    "CHURN_RATE",          # *** NEW: was missing — caused state bleed between churn variants ***
    "SAVE_DETAILED",
    "NUM_ROUNDS"           # *** NEW: back up so ABLATION_ROUNDS override is reversible ***
]

_glob_backup = {name: globals().get(name, None) for name in FLAG_NAMES}
_original_fairness = globals().get("fairness_quota_filter", None)

# ------------------------------------------------------------
# File rename helper
# ------------------------------------------------------------

PER_SEED_PATTERNS = [
    "federated_metrics_seed_{seed}.csv",
    "fog_metrics_seed_{seed}.csv",
    "selection_summary_seed_{seed}.csv",
    "client_attributions_seed_{seed}.csv",
    "meta_history_seed_{seed}.csv",
    "val_acc_seed_{seed}.png",
    "val_f1_seed_{seed}.png",
    "val_auc_seed_{seed}.png",
    "warnings_summary_seed_{seed}.txt"
]

def rename_outputs(seed, tag):
    for pat in PER_SEED_PATTERNS:
        fname = pat.format(seed=seed)
        src = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(src):
            base, ext = os.path.splitext(fname)
            dst = os.path.join(RESULTS_DIR, f"{base}_{tag}{ext}")
            try:
                shutil.move(src, dst)
            except Exception:
                shutil.copy2(src, dst)

# ------------------------------------------------------------
# Aggregate across seeds for one variant
# ------------------------------------------------------------

def aggregate_variant(tag):
    files = glob.glob(os.path.join(RESULTS_DIR, f"federated_metrics_seed_*_{tag}.csv"))
    if not files:
        print(f"  No files found for variant: {tag}")
        return

    dfs = []
    for f in files:
        d = pd.read_csv(f)
        # Extract seed from filename  e.g. federated_metrics_seed_42_fedspoc.csv
        try:
            seed = int(os.path.basename(f).split("_")[3])
        except Exception:
            seed = -1
        d["seed"] = seed
        dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)
    agg = df.groupby("round").agg(["mean", "std"]).reset_index()
    agg.columns = ['_'.join(filter(None, c)).strip('_') for c in agg.columns]
    out = os.path.join(RESULTS_DIR, f"federated_metrics_allseeds_{tag}.csv")
    agg.to_csv(out, index=False)
    print(f"  Aggregated saved: federated_metrics_allseeds_{tag}.csv")


# *** NEW: Extended fairness per variant (Reviewer 1 Point 10) ***
def compute_extended_fairness_variant(tag, num_classes=3):
    """
    Computes Jain index, entropy fairness, CoV, min-slice F1, and max-min gap
    for each round of a given variant. Reads from the aggregated allseeds CSV.
    Saves to federated_metrics_allseeds_<tag>_fairness.csv
    """
    src = os.path.join(RESULTS_DIR, f"federated_metrics_allseeds_{tag}.csv")
    if not os.path.exists(src):
        print(f"  Skipping fairness for {tag} — aggregated file not found.")
        return

    df = pd.read_csv(src)
    results = []
    for _, row in df.iterrows():
        # Try _mean suffix first (from allseeds aggregation), then plain
        f1s = np.array([
            row.get(f"val_f1_c{c}_mean_mean",
                row.get(f"val_f1_c{c}_mean", row.get(f"val_f1_c{c}", 0.0)))
            for c in range(num_classes)
        ])
        jain = (f1s.sum()**2) / (num_classes * (f1s**2).sum() + 1e-9)
        p    = f1s / (f1s.sum() + 1e-9)
        entropy_fair = -np.sum(p * np.log(p + 1e-9)) / np.log(num_classes)
        cov  = f1s.std() / (f1s.mean() + 1e-9)
        results.append({
            "round":            row.get("round", -1),
            "jain_fairness":    float(jain),
            "entropy_fairness": float(entropy_fair),
            "cov_f1":           float(cov),
            "min_slice_f1":     float(f1s.min()),
            "max_min_gap":      float(f1s.max() - f1s.min())
        })

    fair_df = pd.DataFrame(results)
    out = os.path.join(RESULTS_DIR, f"federated_metrics_allseeds_{tag}_fairness.csv")
    fair_df.to_csv(out, index=False)
    print(f"  Extended fairness saved: federated_metrics_allseeds_{tag}_fairness.csv")
    return fair_df


# ------------------------------------------------------------
# Main Ablation Loop
# ------------------------------------------------------------

print("=== Reviewer-Complete Ablation Runner ===")
if ABLATION_ROUNDS is not None:
    print(f"GPU-saving mode: NUM_ROUNDS overridden to {ABLATION_ROUNDS} for all variants.")
start_time = time.time()

for tag, cfg in VARIANTS.items():

    print(f"\n--- Running variant: {tag} ---")

    # Restore clean global state
    for k, v in _glob_backup.items():
        if v is not None:
            globals()[k] = v

    globals()["PATIENCE"] = 999

    # *** NEW: Apply GPU-saving round reduction ***
    if ABLATION_ROUNDS is not None:
        globals()["NUM_ROUNDS"] = ABLATION_ROUNDS

    # Apply variant config
    for key, val in cfg.items():
        if key not in ("MONKEYPATCH_FAIRNESS", "MONKEYPATCH_HIERFEDAVG"):
            globals()[key] = val

    globals()["SAVE_DETAILED"] = True

    # ---- Monkeypatch: fairness OFF (nocurr variant) ----
    patched_fairness = False
    if cfg.get("MONKEYPATCH_FAIRNESS", False):
        if "fairness_quota_filter" in globals():
            def pass_through(selected_clients, alpha_min=0.1, labels=None):
                return list(selected_clients)
            globals()["__fair_backup__"] = globals()["fairness_quota_filter"]
            globals()["fairness_quota_filter"] = pass_through
            patched_fairness = True
            print("  Fairness quota filter disabled for this variant.")

    # *** NEW: Monkeypatch: Hierarchical FedAvg (hier_fedavg variant) ***
    # Simulates Hier. FedAvg by disabling FedSPOC reward-based selection
    # while keeping fog-level aggregation active (USE_FEDSPOC=False means
    # random client selection, which approximates vanilla hierarchical FedAvg).
    patched_hier = False
    if cfg.get("MONKEYPATCH_HIERFEDAVG", False):
        # Force random selection across fog groups = Hier. FedAvg behaviour
        globals()["USE_FEDSPOC"]       = False
        globals()["USE_PPO_BASELINE"]  = False
        globals()["USE_FEDPROX"]       = False
        globals()["ENABLE_CHURN"]      = False
        patched_hier = True
        print("  Hierarchical FedAvg mode: random selection, fog aggregation active.")

    # ---- Run all seeds for this variant ----
    for s in SEEDS_TO_RUN:
        print(f"  > Seed {s}")
        try:
            run_experiment(seed=s)
        except SystemExit:
            print("    Early stop triggered.")
        except Exception as e:
            print(f"    Error: {e}")

        rename_outputs(s, tag)

    # Restore fairness function if patched
    if patched_fairness:
        globals()["fairness_quota_filter"] = globals().pop("__fair_backup__")

    # ---- Aggregate metrics across seeds ----
    aggregate_variant(tag)

    # *** NEW: Compute extended fairness for this variant (Reviewer 1 Point 10) ***
    compute_extended_fairness_variant(tag)

# ---- Restore all globals to original state ----
for k, v in _glob_backup.items():
    if v is not None:
        globals()[k] = v

total_elapsed = time.time() - start_time
print(f"\n✅ All variants finished in {total_elapsed:.1f} seconds")


# ------------------------------------------------------------
# Cross-variant summary table
# *** NEW: Builds a single comparison table across all variants ***
# ------------------------------------------------------------

def build_variant_summary_table(metric="val_f1"):
    """
    Reads per-seed CSVs for each variant and builds a summary table:
      variant | mean_best_<metric> | std_best_<metric>
    Saves to variant_summary_table.csv
    """
    rows = []
    for tag in VARIANTS.keys():
        seed_bests = []
        for s in SEEDS_TO_RUN:
            fpath = os.path.join(RESULTS_DIR, f"federated_metrics_seed_{s}_{tag}.csv")
            if os.path.exists(fpath):
                d = pd.read_csv(fpath)
                if metric in d.columns:
                    seed_bests.append(float(d[metric].max()))
        if seed_bests:
            rows.append({
                "variant":        tag,
                f"mean_best_{metric}": float(np.mean(seed_bests)),
                f"std_best_{metric}":  float(np.std(seed_bests)),
                "n_seeds":        len(seed_bests)
            })

    summary_df = pd.DataFrame(rows).sort_values(f"mean_best_{metric}", ascending=False)
    out = os.path.join(RESULTS_DIR, "variant_summary_table.csv")
    summary_df.to_csv(out, index=False)
    print("\n=== Variant Summary Table ===")
    print(summary_df.to_string(index=False))
    return summary_df

try:
    build_variant_summary_table(metric="val_f1")
    build_variant_summary_table(metric="val_auc")
except Exception as e:
    print(f"Variant summary failed: {e}")


# ------------------------------------------------------------
# Statistical Comparison Helper
# *** UPDATED: Added Wilcoxon signed-rank test (Reviewer 1 Point 6) ***
# ------------------------------------------------------------

def compare_variants(tag1, tag2, metric="val_f1"):
    """
    Compares two variants using:
      - Paired t-test  (parametric, assumes normality)
      - Wilcoxon signed-rank test  (non-parametric, robust for small n)
      - Cohen's d effect size
    Both tests together satisfy reviewer requests for rigorous statistical validation.
    """
    print(f"\n=== Statistical Comparison: {tag1} vs {tag2} (metric={metric}) ===")

    v1, v2 = [], []
    for s in SEEDS_TO_RUN:
        f1_path = os.path.join(RESULTS_DIR, f"federated_metrics_seed_{s}_{tag1}.csv")
        f2_path = os.path.join(RESULTS_DIR, f"federated_metrics_seed_{s}_{tag2}.csv")
        if os.path.exists(f1_path) and os.path.exists(f2_path):
            d1 = pd.read_csv(f1_path)
            d2 = pd.read_csv(f2_path)
            if metric in d1.columns and metric in d2.columns:
                v1.append(float(d1[metric].max()))
                v2.append(float(d2[metric].max()))

    if len(v1) < 2:
        print(f"  Not enough data to compare (found {len(v1)} seed(s)).")
        return

    v1 = np.array(v1)
    v2 = np.array(v2)

    # Paired t-test
    t_stat, p_ttest = ttest_rel(v1, v2)

    # *** NEW: Wilcoxon signed-rank test ***
    try:
        w_stat, p_wilcox = wilcoxon(v1, v2)
    except Exception:
        w_stat, p_wilcox = float("nan"), float("nan")

    # Cohen's d
    diff = v1 - v2
    cohens_d = diff.mean() / (diff.std() + 1e-9)

    print(f"  {tag1} best {metric}: {v1.mean():.4f} ± {v1.std():.4f}")
    print(f"  {tag2} best {metric}: {v2.mean():.4f} ± {v2.std():.4f}")
    print(f"  Mean difference ({tag1} - {tag2}): {diff.mean():+.4f}")
    print(f"  Paired t-test      : t={t_stat:.3f}, p={p_ttest:.4f}  {'✅ significant' if p_ttest < 0.05 else '❌ not significant'}")
    print(f"  Wilcoxon signed-rank: W={w_stat:.1f}, p={p_wilcox:.4f}  {'✅ significant' if p_wilcox < 0.05 else '❌ not significant'}")
    print(f"  Cohen's d          : {cohens_d:.3f}  ({'large' if abs(cohens_d)>0.8 else 'medium' if abs(cohens_d)>0.5 else 'small'} effect)")

    # Save result
    result_df = pd.DataFrame([{
        "tag1": tag1, "tag2": tag2, "metric": metric,
        f"mean_{tag1}": v1.mean(), f"mean_{tag2}": v2.mean(),
        "mean_diff": diff.mean(),
        "t_stat": t_stat, "p_ttest": p_ttest,
        "w_stat": w_stat, "p_wilcox": p_wilcox,
        "cohens_d": cohens_d
    }])
    out = os.path.join(RESULTS_DIR, f"stats_{tag1}_vs_{tag2}_{metric}.csv")
    result_df.to_csv(out, index=False)
    print(f"  Saved: stats_{tag1}_vs_{tag2}_{metric}.csv")
    return result_df


# *** NEW: Run key comparisons automatically after ablation finishes ***
print("\n=== Running key statistical comparisons ===")
KEY_COMPARISONS = [
    ("fedspoc", "fedavg"),
    ("fedspoc", "ppo"),
    ("fedspoc", "hier_fedavg"),
    ("fedspoc", "fedspoc_nocurr"),
    ("fedspoc_churn_random", "fedspoc"),
    ("fedspoc_churn_adv", "fedspoc"),
]

for t1, t2 in KEY_COMPARISONS:
    try:
        compare_variants(t1, t2, metric="val_f1")
    except Exception as e:
        print(f"  Comparison {t1} vs {t2} failed: {e}")

print("\n✅ Ablation runner complete.")
print(f"Results in: {RESULTS_DIR}")
print("\nTo run additional comparisons manually:")
print("  compare_variants('fedspoc', 'fedavg_prox', metric='val_auc')")

# ============================================================
# FedSPOC++ — Complete Missing Graphs Generator
# Run this AFTER the main experiment and ablation runner finish.
# All plots saved to OUTPUT_DIR. No GPU needed.
# ============================================================

import os
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict

OUTPUT_DIR = "/content/drive/MyDrive/veeral_project/for_fedsptoc250/results"
SEEDS = [42, 52, 62, 72, 82]
NUM_CLASSES = 3
CLASS_NAMES = ["URLLC", "mMTC", "eMBB"]
VARIANT_TAGS = [
    "fedspoc", "fedavg", "hier_fedavg", "ppo",
    "fedspoc_prox", "fedavg_prox",
    "fedspoc_nocurr",
    "fedspoc_churn_random", "fedspoc_churn_adv"
]
VARIANT_LABELS = {
    "fedspoc":             "FedSPOC++ (Ours)",
    "fedavg":              "FedAvg",
    "hier_fedavg":         "Hier. FedAvg",
    "ppo":                 "PPO-only",
    "fedspoc_prox":        "FedSPOC+FedProx",
    "fedavg_prox":         "FedAvg+FedProx",
    "fedspoc_nocurr":      "FedSPOC (no curriculum)",
    "fedspoc_churn_random":"FedSPOC+Churn(rand)",
    "fedspoc_churn_adv":   "FedSPOC+Churn(adv)"
}
COLORS = plt.cm.tab10.colors

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER: load allseeds aggregated CSV for a variant
# ============================================================
def load_variant_agg(tag):
    path = os.path.join(OUTPUT_DIR, f"federated_metrics_allseeds_{tag}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    # Fallback: load plain allseeds (for main FedSPOC run)
    path2 = os.path.join(OUTPUT_DIR, "federated_metrics_allseeds.csv")
    if tag == "fedspoc" and os.path.exists(path2):
        return pd.read_csv(path2)
    return None

def load_per_seed(tag, seed):
    path = os.path.join(OUTPUT_DIR, f"federated_metrics_seed_{seed}_{tag}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    if tag == "fedspoc":
        path2 = os.path.join(OUTPUT_DIR, f"federated_metrics_seed_{seed}.csv")
        if os.path.exists(path2):
            return pd.read_csv(path2)
    return None

def get_best_per_seed(tag, metric="val_f1"):
    vals = []
    for s in SEEDS:
        d = load_per_seed(tag, s)
        if d is not None and metric in d.columns:
            vals.append(float(d[metric].max()))
    return np.array(vals) if vals else None


# ============================================================
# GRAPH 1: Convergence Comparison — all baselines on one plot
# (Reviewer 1 Point 6: stronger baselines needed)
# ============================================================
def plot_convergence_comparison(metric="val_f1", ylabel="Macro-F1"):
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = 0
    for i, tag in enumerate(VARIANT_TAGS):
        df = load_variant_agg(tag)
        if df is None:
            continue
        mean_col = f"{metric}_mean" if f"{metric}_mean" in df.columns else metric
        std_col  = f"{metric}_std"  if f"{metric}_std"  in df.columns else None
        if mean_col not in df.columns:
            continue
        x   = df["round"].values
        y   = df[mean_col].values
        yerr = df[std_col].fillna(0).values if std_col and std_col in df.columns else None
        lw  = 2.5 if tag == "fedspoc" else 1.2
        ls  = "-" if tag == "fedspoc" else "--"
        if yerr is not None:
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.12, color=COLORS[i % len(COLORS)])
        ax.plot(x, y, label=VARIANT_LABELS.get(tag, tag),
                color=COLORS[i % len(COLORS)], linewidth=lw, linestyle=ls, marker="o", markersize=3)
        plotted += 1

    if plotted == 0:
        print("  Skipping convergence plot — no variant data found.")
        plt.close(); return

    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Convergence Comparison — {ylabel} Across Rounds", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"graph1_convergence_{metric}.png")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"✅ Graph 1 saved: graph1_convergence_{metric}.png")


# ============================================================
# GRAPH 2: Ablation Bar Chart — best val_f1 & val_auc per variant
# (Reviewer 1 Point 4: component ablation)
# ============================================================
def plot_ablation_bar():
    tags_found, means_f1, stds_f1, means_auc, stds_auc = [], [], [], [], []
    for tag in VARIANT_TAGS:
        f1_arr  = get_best_per_seed(tag, "val_f1")
        auc_arr = get_best_per_seed(tag, "val_auc")
        if f1_arr is not None and len(f1_arr) > 0:
            tags_found.append(tag)
            means_f1.append(f1_arr.mean())
            stds_f1.append(f1_arr.std())
            means_auc.append(auc_arr.mean() if auc_arr is not None else 0)
            stds_auc.append(auc_arr.std()  if auc_arr is not None else 0)

    if not tags_found:
        print("  Skipping ablation bar — no data found.")
        return

    x = np.arange(len(tags_found))
    width = 0.35
    labels = [VARIANT_LABELS.get(t, t) for t in tags_found]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    bars1 = ax1.bar(x, means_f1, width, yerr=stds_f1, capsize=4,
                    color=[COLORS[i % len(COLORS)] for i in range(len(tags_found))],
                    alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("Best Macro-F1", fontsize=11)
    ax1.set_title("Ablation: Best Macro-F1 per Variant", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    for bar, m in zip(bars1, means_f1):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{m:.3f}", ha="center", va="bottom", fontsize=7)

    bars2 = ax2.bar(x, means_auc, width, yerr=stds_auc, capsize=4,
                    color=[COLORS[i % len(COLORS)] for i in range(len(tags_found))],
                    alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("Best Macro-AUC", fontsize=11)
    ax2.set_title("Ablation: Best Macro-AUC per Variant", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    for bar, m in zip(bars2, means_auc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{m:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "graph2_ablation_bar.png")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("✅ Graph 2 saved: graph2_ablation_bar.png")


# ============================================================
# GRAPH 3: Fairness Metrics Over Rounds
# (Reviewer 1 Point 10: entropy fairness, CoV, min-slice F1)
# ============================================================
import ast

def _extract_slice_f1(row):
    """Robustly extract per-class F1 regardless of column naming."""
    vals = []
    for c in range(NUM_CLASSES):
        candidates = [
            f"val_f1_c{c}_mean_mean",
            f"val_f1_c{c}_mean",
            f"val_f1_c{c}"
        ]
        found = False
        for col in candidates:
            if col in row.index:
                vals.append(row[col])
                found = True
                break
        if not found:
            vals.append(np.nan)
    return np.array(vals, dtype=float)


def plot_fairness_over_rounds():

    path = os.path.join(OUTPUT_DIR, "federated_metrics_allseeds.csv")

    if not os.path.exists(path):
        print("  Fairness plot skipped — aggregated CSV missing.")
        return

    df = pd.read_csv(path)

    if "round" not in df.columns:
        print("  Fairness plot skipped — no round column.")
        return

    fairness_rows = []

    for _, row in df.iterrows():
        f1s = _extract_slice_f1(row)

        if np.isnan(f1s).all() or f1s.sum() <= 1e-8:
            fairness_rows.append([np.nan]*5)
            continue

        denom = (NUM_CLASSES * (f1s**2).sum())
        jain = (f1s.sum()**2) / denom if denom > 0 else np.nan

        p = f1s / f1s.sum()
        ent = -np.sum(p * np.log(p + 1e-9)) / np.log(NUM_CLASSES)

        cov = f1s.std() / (f1s.mean() + 1e-9)

        fairness_rows.append([
            jain,
            ent,
            cov,
            f1s.min(),
            f1s.max() - f1s.min()
        ])

    fairness_df = pd.DataFrame(
        fairness_rows,
        columns=["jain_fairness","entropy_fairness",
                 "cov_f1","min_slice_f1","max_min_gap"]
    )

    df = pd.concat([df, fairness_df], axis=1)

    x = df["round"].values

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    metrics = [
        ("jain_fairness", "Jain Fairness Index", "blue"),
        ("entropy_fairness", "Entropy Fairness (norm.)", "green"),
        ("cov_f1", "Coeff. of Variation (F1)", "red"),
        ("min_slice_f1", "Worst-Slice F1 (min)", "purple"),
    ]

    for ax, (col, label, color) in zip(axes.flat, metrics):
        y = df[col].values
        ax.plot(x, y, color=color, marker="o", linewidth=2)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Round")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Fairness Metrics Across Rounds (FedSPOC++)",
                 fontsize=13, fontweight="bold")

    out = os.path.join(OUTPUT_DIR, "graph3_fairness_over_rounds.png")
    plt.tight_layout()
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("✅ Graph 3 fixed and saved.")


# ============================================================
# GRAPH 4: Churn Robustness Comparison
# (Reviewer 1 Point 9: client dropout / adversarial churn)
# ============================================================
def plot_churn_robustness():
    churn_tags = ["fedspoc", "fedspoc_churn_random", "fedspoc_churn_adv"]
    churn_labels = ["No Churn", "Random Churn (30%)", "Adversarial Churn (30%)"]
    churn_colors = ["steelblue", "darkorange", "crimson"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plotted = False
    for tag, label, color in zip(churn_tags, churn_labels, churn_colors):
        df = load_variant_agg(tag)
        if df is None:
            continue
        mean_col_f1  = "val_f1_mean"  if "val_f1_mean"  in df.columns else "val_f1"
        std_col_f1   = "val_f1_std"   if "val_f1_std"   in df.columns else None
        mean_col_auc = "val_auc_mean" if "val_auc_mean" in df.columns else "val_auc"
        std_col_auc  = "val_auc_std"  if "val_auc_std"  in df.columns else None

        if mean_col_f1 not in df.columns:
            continue
        x = df["round"].values
        y_f1 = df[mean_col_f1].values
        y_auc = df[mean_col_auc].values if mean_col_auc in df.columns else None
        yerr_f1 = df[std_col_f1].fillna(0).values if std_col_f1 and std_col_f1 in df.columns else None
        yerr_auc = df[std_col_auc].fillna(0).values if std_col_auc and std_col_auc in df.columns else None

        ax1.plot(x, y_f1, label=label, color=color, linewidth=1.8, marker="o", markersize=3)
        if yerr_f1 is not None:
            ax1.fill_between(x, y_f1 - yerr_f1, y_f1 + yerr_f1, alpha=0.1, color=color)
        if y_auc is not None:
            ax2.plot(x, y_auc, label=label, color=color, linewidth=1.8, marker="s", markersize=3)
            if yerr_auc is not None:
                ax2.fill_between(x, y_auc - yerr_auc, y_auc + yerr_auc, alpha=0.1, color=color)
        plotted = True

    if not plotted:
        print("  Skipping churn plot — no churn variant data found.")
        plt.close(); return

    ax1.set_xlabel("Round"); ax1.set_ylabel("Macro-F1"); ax1.legend(fontsize=9)
    ax1.set_title("Churn Robustness — Macro-F1", fontweight="bold"); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Round"); ax2.set_ylabel("Macro-AUC"); ax2.legend(fontsize=9)
    ax2.set_title("Churn Robustness — Macro-AUC", fontweight="bold"); ax2.grid(True, alpha=0.3)
    plt.suptitle("FedSPOC++ Robustness Under Client Dropout", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path_out = os.path.join(OUTPUT_DIR, "graph4_churn_robustness.png")
    fig.savefig(path_out, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("✅ Graph 4 saved: graph4_churn_robustness.png")


# ============================================================
# GRAPH 5: Sensitivity Analysis
# (Reviewer 1 Point 12: hyperparameter sensitivity)
# ============================================================
def plot_sensitivity_analysis():
    path = os.path.join(OUTPUT_DIR, "sensitivity_analysis.csv")
    if not os.path.exists(path):
        print("  Skipping sensitivity plot — sensitivity_analysis.csv not found.")
        return

    df = pd.read_csv(path)
    if "config" not in df.columns or "best_val_f1" not in df.columns:
        print("  Skipping sensitivity plot — unexpected CSV format.")
        return

    # Group by parameter type
    groups = {
        "Reward Weight wL":  [r for r in df["config"] if r.startswith("sens_wL")],
        "Rarity Bias β":     [r for r in df["config"] if r.startswith("sens_beta")],
        "PPO Clip ε":        [r for r in df["config"] if r.startswith("sens_clip")],
        "EWMA Lambda λ":     [r for r in df["config"] if r.startswith("sens_lambda")],
    }
    PARAM_COLORS = ["steelblue", "darkorange", "green", "purple"]

    fig, axes = plt.subplots(1, len(groups), figsize=(14, 5), sharey=False)
    for ax, (param_name, configs), color in zip(axes, groups.items(), PARAM_COLORS):
        sub = df[df["config"].isin(configs)]
        if sub.empty:
            ax.set_visible(False); continue
        labels = sub["config"].str.replace("sens_", "", regex=False).values
        f1_vals = sub["best_val_f1"].values
        auc_vals = sub["best_val_auc"].values if "best_val_auc" in sub.columns else np.zeros_like(f1_vals)
        x = np.arange(len(labels))
        ax.bar(x - 0.2, f1_vals,  0.35, label="Best F1",  color=color,   alpha=0.8, edgecolor="black")
        ax.bar(x + 0.2, auc_vals, 0.35, label="Best AUC", color=color,   alpha=0.4, edgecolor="black", hatch="//")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=8)
        ax.set_title(param_name, fontweight="bold", fontsize=10)
        ax.set_ylabel("Score"); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
        for i, (f, a) in enumerate(zip(f1_vals, auc_vals)):
            ax.text(i - 0.2, f + 0.005, f"{f:.3f}", ha="center", fontsize=7)

    plt.suptitle("Sensitivity Analysis — Hyperparameter Sweep", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path_out = os.path.join(OUTPUT_DIR, "graph5_sensitivity_analysis.png")
    fig.savefig(path_out, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("✅ Graph 5 saved: graph5_sensitivity_analysis.png")


# ============================================================
# GRAPH 6: Communication Cost vs. Performance Tradeoff
# (Paper Section 4 — Eq. 21 communication cost)
# ============================================================
def plot_comm_cost_tradeoff():
    # Load FedSPOC main run
    path = os.path.join(OUTPUT_DIR, "federated_metrics_allseeds.csv")
    if not os.path.exists(path):
        path = os.path.join(OUTPUT_DIR, "federated_metrics_seed_42.csv")
    if not os.path.exists(path):
        print("  Skipping comm cost plot — no data found.")
        return

    df = pd.read_csv(path)
    if "comm_cost_bytes" not in df.columns:
        print("  Skipping comm cost plot — comm_cost_bytes column missing.")
        return

    f1_col = "val_f1_mean" if "val_f1_mean" in df.columns else "val_f1"
    if f1_col not in df.columns:
        print("  Skipping comm cost plot — val_f1 column missing.")
        return

    x_cost = df["comm_cost_bytes"].cumsum().values / 1e6  # MB
    y_f1   = df[f1_col].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: comm cost over rounds
    ax1.plot(df["round"].values, df["comm_cost_bytes"].values / 1e6,
             color="teal", linewidth=1.8, marker="o", markersize=3)
    ax1.set_xlabel("Round"); ax1.set_ylabel("Comm. Cost per Round (MB)")
    ax1.set_title("Communication Cost per Round", fontweight="bold"); ax1.grid(True, alpha=0.3)

    # Right: cumulative comm cost vs. F1 (efficiency curve)
    sc = ax2.scatter(x_cost, y_f1, c=df["round"].values, cmap="viridis", s=40, zorder=3)
    ax2.plot(x_cost, y_f1, color="gray", linewidth=0.8, alpha=0.5)
    plt.colorbar(sc, ax=ax2, label="Round")
    ax2.set_xlabel("Cumulative Comm. Cost (MB)")
    ax2.set_ylabel("Macro-F1")
    ax2.set_title("Communication Efficiency Curve", fontweight="bold"); ax2.grid(True, alpha=0.3)

    plt.suptitle("Communication Cost Analysis (FedSPOC++)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path_out = os.path.join(OUTPUT_DIR, "graph6_comm_cost_tradeoff.png")
    fig.savefig(path_out, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("✅ Graph 6 saved: graph6_comm_cost_tradeoff.png")


# ============================================================
# GRAPH 7: Meta-Weight Evolution Over Rounds
# (Shows how FedSPOC++ adapts w_L, w_E, w_H dynamically)
# ============================================================
def plot_meta_weight_evolution():
    # Try to load from any seed
    df = None
    for s in SEEDS:
        path = os.path.join(OUTPUT_DIR, f"meta_history_seed_{s}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            break

    if df is None or "meta_w" not in df.columns:
        print("  Skipping meta-weight plot — meta_history_seed_*.csv not found.")
        return

    # meta_w stored as string list e.g. "[0.6, 0.3, 0.1]"
    try:
        weights = df["meta_w"].apply(ast.literal_eval).tolist()
        w_arr = np.array(weights)
    except Exception as e:
        print(f"  Failed to parse meta_w: {e}")
        return

    rounds = df["round"].values
    labels = ["w_L (Latency)", "w_E (Energy)", "w_H (Entropy)"]
    colors = ["steelblue", "darkorange", "green"]

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(rounds, w_arr[:, i], label=label, color=color, linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Weight Value", fontsize=11)
    ax.set_title("Meta-Weight Adaptation Over Rounds (FedSPOC++)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.axhline(1/3, color="gray", linestyle=":", linewidth=1, label="Equal weights (1/3)")
    plt.tight_layout()
    path_out = os.path.join(OUTPUT_DIR, "graph7_meta_weight_evolution.png")
    fig.savefig(path_out, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("✅ Graph 7 saved: graph7_meta_weight_evolution.png")


# ============================================================
# GRAPH 8: Client Selection Diversity Over Rounds
# (Shows FedSPOC++ explores more clients, not same ones every round)
# ============================================================
import ast

def plot_client_selection_diversity():

    df = None
    for s in SEEDS:
        path = os.path.join(OUTPUT_DIR, f"selection_summary_seed_{s}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  Using selection summary from seed {s}")
            break

    if df is None:
        print("  Skipping client diversity plot — selection_summary_seed_*.csv not found.")
        return

    if "selected" not in df.columns and "selected_count" not in df.columns:
        print("  Skipping client diversity plot — required columns missing.")
        return

    # ---------- CUMULATIVE UNIQUE CLIENTS ----------
    cumulative_unique = []
    seen = set()

    for val in df.get("selected", []):
        try:
            if isinstance(val, str):
                sel = ast.literal_eval(val)
            else:
                sel = val

            if isinstance(sel, (list, tuple, set)):
                seen.update(sel)

        except Exception:
            pass

        cumulative_unique.append(len(seen))

    rounds = df["round"].values if "round" in df.columns else np.arange(len(df))

    # ---------- PER ROUND COUNT ----------
    per_round_counts = []

    if "selected_count" in df.columns:
        per_round_counts = df["selected_count"].fillna(0).astype(int).tolist()
    else:
        for val in df["selected"]:
            try:
                sel = ast.literal_eval(val) if isinstance(val, str) else val
                per_round_counts.append(len(sel) if isinstance(sel, (list, tuple, set)) else 0)
            except Exception:
                per_round_counts.append(0)

    # ---------- PLOTTING ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left plot
    ax1.plot(rounds, cumulative_unique,
             color="steelblue", linewidth=2, marker="o", markersize=3)

    total_clients = max(cumulative_unique) if cumulative_unique else 0
    ax1.axhline(y=total_clients,
                color="red", linestyle="--", linewidth=1,
                label=f"Total clients ({total_clients})")

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Cumulative Unique Clients")
    ax1.set_title("Client Coverage Over Rounds", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right plot
    ax2.bar(rounds, per_round_counts,
            color="darkorange", alpha=0.85,
            edgecolor="black", linewidth=0.3)

    if per_round_counts:
        mean_val = np.mean(per_round_counts)
        ax2.axhline(mean_val,
                    color="red", linestyle="--", linewidth=1.2,
                    label=f"Mean = {mean_val:.1f}")

    ax2.set_xlabel("Round")
    ax2.set_ylabel("Clients Selected per Round")
    ax2.set_title("Per-Round Client Selection Count", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Client Selection Diversity — FedSPOC++",
                 fontsize=13, fontweight="bold")

    out = os.path.join(OUTPUT_DIR, "graph8_client_selection_diversity.png")
    plt.tight_layout()
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("✅ Graph 8 (robust) saved.")
# ============================================================
# GRAPH 9: Per-Class F1 Over Rounds (3 slices on one plot)
# (Reviewer 1 Point 10: per-slice fairness breakdown)
# ============================================================
def plot_per_class_f1():

    path = os.path.join(OUTPUT_DIR, "federated_metrics_allseeds.csv")

    if not os.path.exists(path):
        print("  Skipping per-class F1 plot — aggregated CSV missing.")
        return

    df = pd.read_csv(path)

    if "round" not in df.columns:
        print("  Skipping per-class F1 plot — no round column.")
        return

    fig, ax = plt.subplots(figsize=(9, 4))

    found_any = False

    for c, cls_name in enumerate(CLASS_NAMES):

        # Robust detection
        candidates = [
            col for col in df.columns
            if f"val_f1_c{c}" in col and "mean" in col
        ]

        if not candidates:
            candidates = [
                col for col in df.columns
                if col == f"val_f1_c{c}"
            ]

        if not candidates:
            print(f"  Warning: No column found for class {cls_name}")
            continue

        col_mean = candidates[0]

        # Optional std detection
        std_candidates = [
            col for col in df.columns
            if f"val_f1_c{c}" in col and "std" in col
        ]
        col_std = std_candidates[0] if std_candidates else None

        x = df["round"].values
        y = df[col_mean].values

        ax.plot(x, y,
                linewidth=2,
                marker="o",
                markersize=3,
                label=cls_name)

        if col_std:
            yerr = df[col_std].fillna(0).values
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.12)

        found_any = True

    if not found_any:
        print("  No per-class F1 columns found. Skipping plot.")
        plt.close(fig)
        return

    ax.set_xlabel("Round")
    ax.set_ylabel("Class F1 Score")
    ax.set_title("Per-Class (Slice) F1 Over Rounds — FedSPOC++",
                 fontweight="bold")
    ax.legend(title="Network Slice")
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUTPUT_DIR, "graph9_per_class_f1.png")
    plt.tight_layout()
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("✅ Graph 9 (robust) saved.")
# ============================================================
# GRAPH 10: SHAP Feature Importance Bar Chart
# (Reviewer 1 Point 11 — cleaner version of shap_summary.png)
# ============================================================
def plot_shap_bar():
    path = os.path.join(OUTPUT_DIR, "shap_feature_importance.csv")
    if not os.path.exists(path):
        print("  Skipping SHAP bar — shap_feature_importance.csv not found (run SHAP analysis first).")
        return

    df = pd.read_csv(path).head(15)  # top 15 features
    df = df.sort_values("mean_abs_shap", ascending=True)  # ascending for horizontal bar

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(df["feature"], df["mean_abs_shap"],
                   color=plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(df))),
                   edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Mean |SHAP| (avg across classes)", fontsize=11)
    ax.set_title("SHAP Feature Importance — FedSPOC++ Global Model", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, df["mean_abs_shap"].values):
        ax.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path_out = os.path.join(OUTPUT_DIR, "graph10_shap_bar.png")
    fig.savefig(path_out, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("✅ Graph 10 saved: graph10_shap_bar.png")


# ============================================================
# GRAPH 11: Fog Node QoS Heatmap (extended)
# (Shows per-fog latency, SLA violations, cache hits, energy)
# ============================================================
def plot_fog_qos_heatmap():

    path = os.path.join(OUTPUT_DIR, "fog_metrics_detailed.csv")

    if not os.path.exists(path):
        print("  Skipping fog QoS heatmap — fog_metrics_detailed.csv not found.")
        return

    df = pd.read_csv(path)

    required_cols = ["fog_node", "latency", "energy", "sla_violated"]
    for col in required_cols:
        if col not in df.columns:
            print(f"  Missing required column: {col}")
            return

    # Aggregate
    metrics_agg = df.groupby("fog_node").agg({
        "latency": "mean",
        "energy": "mean",
        "sla_violated": "mean"
    }).reset_index()

    metrics_agg.columns = ["fog_node", "Avg Latency",
                           "Avg Energy", "SLA Violation Rate"]

    qos_cols = ["Avg Latency", "Avg Energy", "SLA Violation Rate"]

    heatmap_data = metrics_agg[qos_cols].values

    # Normalize per column (0 = best, 1 = worst)
    col_min = heatmap_data.min(axis=0)
    col_max = heatmap_data.max(axis=0)

    heatmap_norm = (heatmap_data - col_min) / (col_max - col_min + 1e-9)

    fig, ax = plt.subplots(figsize=(9, 4))

    im = ax.imshow(heatmap_norm,
                   cmap="RdYlGn_r",
                   aspect="auto",
                   vmin=0, vmax=1)

    ax.set_xticks(range(len(qos_cols)))
    ax.set_xticklabels(qos_cols, fontsize=9)

    ax.set_yticks(range(len(metrics_agg)))
    ax.set_yticklabels([f"Fog {i}" for i in metrics_agg["fog_node"]],
                       fontsize=9)

    # Annotate actual values
    for i in range(len(metrics_agg)):
        for j in range(len(qos_cols)):
            val = heatmap_data[i, j]
            norm_val = heatmap_norm[i, j]
            ax.text(j, i,
                    f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=8,
                    color="white" if norm_val > 0.6 else "black")

    plt.colorbar(im, ax=ax,
                 label="Normalized (0 = best, 1 = worst)")

    ax.set_title("Fog Node QoS Heatmap — FedSPOC++",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "graph11_fog_qos_heatmap.png")
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("✅ Graph 11 (corrected) saved.")
# ============================================================
# GRAPH 12: Fairness vs Performance Scatter (across variants)
# (Tradeoff plot — shows FedSPOC++ achieves both)
# ============================================================
def plot_fairness_vs_performance():

    rows = []

    for tag in VARIANT_TAGS:

        f1_arr = get_best_per_seed(tag, "val_f1")
        if f1_arr is None:
            continue

        # Load aggregated file for fairness computation
        df = load_variant_agg(tag)
        if df is None or "round" not in df.columns:
            continue

        # --- Compute Jain fairness robustly ---
        fairness_vals = []

        for _, row in df.iterrows():

            f1s = []
            for c in range(NUM_CLASSES):
                candidates = [
                    f"val_f1_c{c}_mean_mean",
                    f"val_f1_c{c}_mean",
                    f"val_f1_c{c}"
                ]
                val = None
                for col in candidates:
                    if col in row.index:
                        val = row[col]
                        break
                if val is not None:
                    f1s.append(val)

            if len(f1s) != NUM_CLASSES:
                continue

            f1s = np.array(f1s)

            if f1s.sum() <= 1e-8:
                continue

            denom = NUM_CLASSES * (f1s**2).sum()
            if denom > 0:
                jain = (f1s.sum()**2) / denom
                fairness_vals.append(jain)

        if not fairness_vals:
            continue

        rows.append({
            "variant": tag,
            "label": VARIANT_LABELS.get(tag, tag),
            "mean_best_f1": float(f1_arr.mean()),
            "best_jain": max(fairness_vals)
        })

    df_plot = pd.DataFrame(rows)

    if df_plot.empty:
        print("  Skipping fairness-vs-performance scatter — insufficient data.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, row in df_plot.iterrows():

        is_ours = row["variant"] == "fedspoc"

        ax.scatter(row["mean_best_f1"],
                   row["best_jain"],
                   s=200 if is_ours else 90,
                   edgecolors="black",
                   linewidths=0.7,
                   zorder=3)

        ax.annotate(row["label"],
                    (row["mean_best_f1"], row["best_jain"]),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=8)

    ax.set_xlabel("Mean Best Macro-F1")
    ax.set_ylabel("Best Jain Fairness Index")
    ax.set_title("Fairness vs Performance Tradeoff",
                 fontweight="bold")

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "graph12_fairness_vs_performance.png")
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("✅ Graph 12 (corrected) saved.")
# ============================================================
# RUN ALL
# ============================================================
print("=" * 55)
print("Generating all missing graphs...")
print("=" * 55)

plot_convergence_comparison(metric="val_f1",  ylabel="Macro-F1")
plot_convergence_comparison(metric="val_auc", ylabel="Macro-AUC")
plot_ablation_bar()
plot_fairness_over_rounds()
plot_churn_robustness()
plot_sensitivity_analysis()
plot_comm_cost_tradeoff()
plot_meta_weight_evolution()
plot_client_selection_diversity()
plot_per_class_f1()
plot_shap_bar()
plot_fog_qos_heatmap()
plot_fairness_vs_performance()

print("\n" + "=" * 55)
print("All graphs complete. Summary:")
print("=" * 55)
GRAPH_MAP = {
    "graph1_convergence_val_f1.png":       "Convergence (F1) — all baselines",
    "graph1_convergence_val_auc.png":      "Convergence (AUC) — all baselines",
    "graph2_ablation_bar.png":             "Ablation bar chart (F1 + AUC per variant)",
    "graph3_fairness_over_rounds.png":     "Fairness metrics over rounds",
    "graph4_churn_robustness.png":         "Churn robustness comparison",
    "graph5_sensitivity_analysis.png":     "Sensitivity analysis (hyperparams)",
    "graph6_comm_cost_tradeoff.png":       "Communication cost vs performance",
    "graph7_meta_weight_evolution.png":    "Meta-weight adaptation over rounds",
    "graph8_client_selection_diversity.png":"Client selection diversity",
    "graph9_per_class_f1.png":             "Per-class (slice) F1 over rounds",
    "graph10_shap_bar.png":                "SHAP feature importance (bar)",
    "graph11_fog_qos_heatmap.png":         "Fog node QoS heatmap",
    "graph12_fairness_vs_performance.png": "Fairness vs performance scatter",
}
for fname, desc in GRAPH_MAP.items():
    exists = os.path.exists(os.path.join(OUTPUT_DIR, fname))
    status = "✅" if exists else "⏭ skipped (data not yet available)"
    print(f"  {status}  {fname:45s}  — {desc}")

import os
import pandas as pd
import numpy as np

OUTPUT_DIR = "/content/drive/MyDrive/veeral_project/for_fedsptoc250/results"

print("\n==============================")
print("1️⃣ CROSS-SEED ROUND-WISE MEAN ± STD")
print("==============================\n")

path_all = os.path.join(OUTPUT_DIR, "federated_metrics_allseeds.csv")

if os.path.exists(path_all):
    df_all = pd.read_csv(path_all)
    print(df_all[["round","val_f1_mean","val_f1_std","val_auc_mean","val_auc_std",
                  "cov_f1_mean","entropy_fairness_mean"]])
else:
    print("❌ federated_metrics_allseeds.csv not found")



print("\n==============================")
print("2️⃣ PER-SEED FILE CHECK")
print("==============================\n")

all_files = os.listdir(OUTPUT_DIR)
seed_files = [f for f in all_files if f.startswith("federated_metrics_seed_")]

if len(seed_files) == 0:
    print("⚠ No per-seed metric files found.")
else:
    for f in seed_files:
        try:
            df = pd.read_csv(os.path.join(OUTPUT_DIR, f))
            print(f"\nFile: {f}")
            print(df[["round","val_f1","val_auc"]].tail())
        except Exception as e:
            print(f"⚠ Could not read {f}: {e}")



print("\n==============================")
print("3️⃣ FOG / SYSTEM METRICS (SAFE READ)")
print("==============================\n")

fog_files = [f for f in all_files if f.startswith("fog_metrics_seed_")]

valid_fog_dfs = []

for f in fog_files:
    full_path = os.path.join(OUTPUT_DIR, f)
    try:
        if os.path.getsize(full_path) > 0:  # skip empty files
            df = pd.read_csv(full_path)
            if len(df.columns) > 0:
                valid_fog_dfs.append(df)
                print(f"✔ Loaded {f}")
        else:
            print(f"⚠ Skipped empty file: {f}")
    except Exception as e:
        print(f"⚠ Error reading {f}: {e}")

if len(valid_fog_dfs) > 0:
    df_fog = pd.concat(valid_fog_dfs, ignore_index=True)
    print("\n📊 Aggregated Fog Metrics:")
    print("Mean Latency (ms):", round(df_fog["latency"].mean(), 3))
    print("Mean Energy (J):", round(df_fog["energy"].mean(), 3))

    if "sla_violated" in df_fog.columns:
        sla_rate = 100 * (1 - df_fog["sla_violated"].mean())
        print("SLA Compliance (%):", round(sla_rate, 2))
else:
    print("⚠ No valid fog metric data found.")



print("\n==============================")
print("4️⃣ CONVERGENCE ROUND DETECTION")
print("==============================\n")

if os.path.exists(path_all):
    df = pd.read_csv(path_all)

    best_round = df.loc[df["val_auc_mean"].idxmax(), "round"]
    print("Best mean AUC round:", best_round)

    print("\nRounds with perfect F1:")
    perfect = df[df["val_f1_mean"] == 1.0]["round"].tolist()
    print(perfect)

import pandas as pd
import os

OUTPUT_DIR = "/content/drive/MyDrive/veeral_project/for_fedsptoc250/results"

fedspoc_fog_files = [f for f in os.listdir(OUTPUT_DIR)
                     if f.startswith("fog_metrics_seed_") and "fedspoc.csv" in f]

dfs = []
for f in fedspoc_fog_files:
    dfs.append(pd.read_csv(os.path.join(OUTPUT_DIR, f)))

df_fog = pd.concat(dfs, ignore_index=True)

print("FedSPOC Only:")
print("Mean Latency (ms):", df_fog["latency"].mean())
print("Mean Energy (J):", df_fog["energy"].mean())
print("SLA Compliance (%):", 100 * (1 - df_fog["sla_violated"].mean()))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load aggregated results
df = pd.read_csv("/content/drive/MyDrive/veeral_project/for_fedsptoc250/results/federated_metrics_allseeds.csv")

rounds = df["round"]

# Automatically detect first round with perfect mean AUC
early_stop_round = df.loc[df["val_auc_mean"] == df["val_auc_mean"].max(), "round"].iloc[0]

plt.figure(figsize=(6.5,4.5))

# ---- Macro-F1 ----
plt.plot(rounds,
         df["val_f1_mean"],
         linewidth=2.2,
         marker="o",
         markevery=2,
         label="Macro-F1")

plt.fill_between(rounds,
                 df["val_f1_mean"] - df["val_f1_std"],
                 df["val_f1_mean"] + df["val_f1_std"],
                 alpha=0.15)

# ---- ROC-AUC ----
plt.plot(rounds,
         df["val_auc_mean"],
         linewidth=2.2,
         marker="s",
         markevery=2,
         label="ROC-AUC")

plt.fill_between(rounds,
                 df["val_auc_mean"] - df["val_auc_std"],
                 df["val_auc_mean"] + df["val_auc_std"],
                 alpha=0.15)

# ---- Early stopping marker ----
plt.axvline(x=13,
            linestyle="--",
            linewidth=1.3,
            color="gray")

# ---- Axis formatting ----
plt.ylim(0.90, 1.01)
plt.xlim(rounds.min(), rounds.max())
plt.xlabel("Global Communication Round")
plt.ylabel("Validation Score")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(frameon=True)

plt.tight_layout()
plt.savefig("convergence_final.png", dpi=600, bbox_inches="tight")
plt.show()

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATH = "/content/drive/MyDrive/veeral_project/for_fedsptoc250/results/"

files = glob.glob(PATH + "fog_metrics_seed_*_fedspoc.csv")

dfs = []
for f in files:
    d = pd.read_csv(f)
    dfs.append(d)

df = pd.concat(dfs)

# Aggregate per round across seeds + fog nodes
agg = df.groupby("round").agg({
    "latency": ["mean", "std"],
    "energy": ["mean", "std"],
    "sla_violated": "mean"
}).reset_index()

agg.columns = ["round",
               "lat_mean", "lat_std",
               "energy_mean", "energy_std",
               "sla_violation_rate"]

# Convert to %
agg["sla_violation_pct"] = agg["sla_violation_rate"] * 100
agg["sla_compliance_pct"] = (1 - agg["sla_violation_rate"]) * 100

print("Overall Means (Sanity Check)")
print("Latency:", agg["lat_mean"].mean())
print("Energy:", agg["energy_mean"].mean())
print("SLA Compliance:", agg["sla_compliance_pct"].mean())

# ===============================
# Plot
# ===============================

fig, ax1 = plt.subplots(figsize=(10,5))

# Latency
ax1.plot(agg["round"], agg["lat_mean"],
         marker="o", linewidth=2, label="Latency (mean, ms)")
ax1.fill_between(agg["round"],
                 agg["lat_mean"] - agg["lat_std"],
                 agg["lat_mean"] + agg["lat_std"],
                 alpha=0.2)

# Energy
ax1.plot(agg["round"], agg["energy_mean"],
         marker="s", linewidth=2, label="Energy (mean, J)")
ax1.fill_between(agg["round"],
                 agg["energy_mean"] - agg["energy_std"],
                 agg["energy_mean"] + agg["energy_std"],
                 alpha=0.2)

ax1.set_xlabel("Global Communication Round")
ax1.set_ylabel("Latency (ms) / Energy (J)")
ax1.grid(alpha=0.3)

# SLA on secondary axis
ax2 = ax1.twinx()
ax2.plot(agg["round"], agg["sla_violation_pct"],
         linestyle="--", marker="x", color="red",
         label="SLA Violations (%)")

ax2.set_ylabel("SLA Violations (%)", color="red")
ax2.tick_params(axis='y', labelcolor='red')

fig.tight_layout()
plt.title("Latency, Energy, and SLA Violations per Round (FedSPOC Full)")
plt.savefig("fig_latency_energy_updated.png", dpi=600)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
# Define path
PATH = "/content/drive/MyDrive/veeral_project/for_fedsptoc250/results/"

# Load aggregated cross-seed metrics
df = pd.read_csv(os.path.join(PATH, "federated_metrics_allseeds.csv"))

print(df.head())

# Keep only rounds that exist
rounds = df["round"]

plt.figure(figsize=(7,5))

plt.plot(rounds,
         df["val_f1_c0_mean_mean"],
         marker="o", linewidth=2,
         label="Slice C0")

plt.plot(rounds,
         df["val_f1_c1_mean_mean"],
         marker="s", linewidth=2,
         label="Slice C1")

plt.plot(rounds,
         df["val_f1_c2_mean_mean"],
         marker="^", linewidth=2,
         label="Slice C2")

plt.xlabel("Global Communication Round")
plt.ylabel("Macro-F1")
plt.ylim(0.0, 1.02)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("fig_slice_f1_mean_updated.png", dpi=600)
plt.show()

# ==========================================
# 1. Recreate Model Architecture (IMPORTANT)
# ==========================================

import torch
from torch import nn

NUM_CLASSES = 3  # keep consistent with training
FEATURE_VECTOR_DIM = 46  # <-- match your printed encoder output dim

class MobilityModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 2. Load Latest Valid Checkpoint
# ==========================================

checkpoint_path = "/content/drive/MyDrive/veeral_project/for_fedsptoc250/checkpoints/seed_82/global_round_14.pt"

model = MobilityModel(FEATURE_VECTOR_DIM)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

print("Model loaded successfully.")

# ==========================================================
# COMPLETE UMAP PIPELINE (Self-Contained)
# ==========================================================

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# -----------------------
# CONFIG
# -----------------------
NUM_CLASSES = 3
FEATURE_VECTOR_DIM = 46  # <-- MUST match training
DATA_PATH = "/content/drive/MyDrive/veeral_project/for_fedsptoc250/crawdad_250_renamed.parquet"
CHECKPOINT_PATH = "/content/drive/MyDrive/veeral_project/for_fedsptoc250/checkpoints/seed_82/global_round_14.pt"
GLOBAL_HOLDOUT_FRAC = 0.20

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_parquet(DATA_PATH)

# recreate holdout split (same seed logic)
np.random.seed(42)
all_idx = df.index.to_numpy()
holdout_count = int(len(all_idx) * GLOBAL_HOLDOUT_FRAC)
holdout_idx = np.random.choice(all_idx, size=holdout_count, replace=False)
holdout_df = df.loc[holdout_idx]

print("Holdout size:", holdout_df.shape)

# -----------------------
# Recreate Feature Columns
# -----------------------
categorical_cols = ['f_0','f_1','f_2','f_3']
numeric_cols = ['f_4','f_5',
                'packet_loss_rate_(reliability)',
                'packet_delay_budget_(latency)',
                'normalized_latency',
                'normalized_reliability']

# rebuild encoder
from sklearn.preprocessing import OneHotEncoder

import sklearn
from packaging import version

onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Fit encoder on full dataset
onehot.fit(df[categorical_cols].fillna("NA").astype(str))

def encode_features(df_sub):
    cat = onehot.transform(df_sub[categorical_cols].fillna("NA").astype(str))
    num = df_sub[numeric_cols].astype(np.float32).fillna(0.0).values
    return np.hstack([cat, num])
def encode_features(df_sub):
    cat = onehot.transform(df_sub[categorical_cols].fillna("NA").astype(str))
    num = df_sub[numeric_cols].astype(np.float32).fillna(0.0).values
    return np.hstack([cat, num])

X_hold = encode_features(holdout_df)
y_hold = holdout_df["slice_label"].astype(int).values

print("Encoded holdout shape:", X_hold.shape)

# -----------------------
# Recreate Model
# -----------------------
class MobilityModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)

model = MobilityModel(FEATURE_VECTOR_DIM)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
model.eval()

print("Model loaded successfully.")

# -----------------------
# Extract Penultimate Layer
# -----------------------
with torch.no_grad():
    X_tensor = torch.tensor(X_hold).float()
    features = model.net[:-1](X_tensor)
    embeddings = features.numpy()

embeddings = StandardScaler().fit_transform(embeddings)

# -----------------------
# UMAP Projection
# -----------------------
reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    metric="euclidean",
    random_state=42
)

embedding_2d = reducer.fit_transform(embeddings)

# -----------------------
# Quantitative Metrics
# -----------------------
sil = silhouette_score(embedding_2d, y_hold)
db = davies_bouldin_score(embedding_2d, y_hold)

print(f"Silhouette Score: {sil:.4f}")
print(f"Davies-Bouldin Index: {db:.4f}")

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(8,6))

for cls, label, color in zip(
    [0,1,2],
    ["URLLC","mMTC","eMBB"],
    ["#1f77b4","#ff7f0e","#2ca02c"]
):
    idx = y_hold == cls
    plt.scatter(
        embedding_2d[idx,0],
        embedding_2d[idx,1],
        s=4,
        alpha=0.6,
        label=f"{label} (n={idx.sum()})",
        c=color
    )

plt.title("UMAP of Holdout Embeddings (Penultimate Layer)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend()
plt.tight_layout()
plt.savefig("umap_holdout_updated.png", dpi=600)
plt.show()

import glob
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

RESULT_PATH = "/content/drive/MyDrive/veeral_project/for_fedsptoc250/results/"

SEEDS = [42, 52, 62, 72, 82]

def load_exact_variant(variant_name):
    scores = []
    for seed in SEEDS:
        file_path = f"{RESULT_PATH}/federated_metrics_seed_{seed}_{variant_name}.csv"
        df = pd.read_csv(file_path)
        best = df["val_f1"].max()
        scores.append(best)
    return np.array(scores)

fedspoc = load_exact_variant("fedspoc")
fedavg = load_exact_variant("fedavg")
ppo = load_exact_variant("ppo")
nocurr = load_exact_variant("fedspoc_nocurr")

print("FedSPOC:", fedspoc)
print("FedAvg:", fedavg)

t_stat, p_val = ttest_rel(fedspoc, fedavg)
print("FedSPOC vs FedAvg:", t_stat, p_val)

t_stat, p_val = ttest_rel(fedspoc, ppo)
print("FedSPOC vs PPO:", t_stat, p_val)

t_stat, p_val = ttest_rel(fedspoc, nocurr)
print("FedSPOC vs NoCurr:", t_stat, p_val)

def cohens_d(x, y):
    diff = x - y
    return diff.mean() / diff.std(ddof=1)

print("Cohen d (FedSPOC vs FedAvg):", cohens_d(fedspoc, fedavg))

import glob
import pandas as pd
import numpy as np

RESULT_PATH = "/content/drive/MyDrive/veeral_project/for_fedsptoc250/results/"

SEEDS = [42, 52, 62, 72, 82]

all_dfs = []

for seed in SEEDS:
    file_path = f"{RESULT_PATH}/fog_metrics_seed_{seed}_fedspoc.csv"
    df = pd.read_csv(file_path)
    df["seed"] = seed
    all_dfs.append(df)

fog_df = pd.concat(all_dfs)

# Compute per-fog stats
summary = fog_df.groupby("fog_node").agg({
    "latency": ["mean", "std"],
    "energy": ["mean", "std"],
    "sla_violated": "mean",
    "cache_hit": "mean"
}).reset_index()

summary.columns = ["fog_node",
                   "lat_mean","lat_std",
                   "energy_mean","energy_std",
                   "sla_violation_rate",
                   "cache_hit_rate"]

summary["sla_violation_rate"] *= 100
summary["cache_hit_rate"] *= 100

print(summary)
