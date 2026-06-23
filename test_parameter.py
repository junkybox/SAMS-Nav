import os


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def _env_int(name, default):
    return int(os.getenv(name, default))


def _env_float(name, default):
    return float(os.getenv(name, default))


## Test Options
TEST_METHOD = 'DARE' # It's only DARE
DATA_TYPE = 'node'  # 'map', 'node'
USE_TEST_DATASET = True  # False for train dataset, True for test dataset
USE_DELTA_POSITION = True # False for absolute position, True for delta position
TEST_N_AGENTS = 1 # SINGLE AGENT keep it 1

## Environment Runner Options
USE_GPU = True
# comma-separated override, e.g. DARE_DEVICE_LIST="cuda:0" or "cuda:0,cuda:1"
DEVICE_LIST = [d.strip() for d in os.getenv('DARE_DEVICE_LIST', 'cuda:1').split(',') if d.strip()]
NUM_META_AGENT = _env_int('DARE_NUM_META_AGENT', 10)  # the number of processes

# Visualization run: 20 unseen episodes with a GIF saved for each. For a
# precise metric sweep instead, set NUM_TEST=1000, SAVE_GIFS=False.
NUM_TEST = _env_int('DARE_NUM_TEST', 1000)
NUM_RUN = 1
SAVE_GIFS = _env_bool('DARE_SAVE_GIFS', True)

ACTION_HORIZON = None # None for 1 horizon

## Candidate Sampling, Ranking, and Safety Layer
# A: original single sample; B: random valid best-of-K; C: goal progress;
# D: full safety-aware weighted ranking.
EXPERIMENT_MODE = os.getenv('DARE_EXPERIMENT_MODE', 'D').upper()
if EXPERIMENT_MODE not in ('A', 'B', 'C', 'D'):
    raise ValueError('DARE_EXPERIMENT_MODE must be A, B, C, or D')
NUM_ACTION_SAMPLES = _env_int('DARE_NUM_SAMPLES', 8)
if NUM_ACTION_SAMPLES < 1:
    raise ValueError('DARE_NUM_SAMPLES must be at least 1')
ENABLE_RANKING = _env_bool(
    'DARE_ENABLE_RANKING', EXPERIMENT_MODE in ('C', 'D')
)
SNAP_RADIUS_FACTOR = _env_float('DARE_SNAP_RADIUS_FACTOR', 0.75)

W_GOAL = _env_float('DARE_W_GOAL', 2.0)
W_CONNECTIVITY = _env_float('DARE_W_CONNECTIVITY', 1.0)
W_UTILITY = _env_float('DARE_W_UTILITY', 0.35)
W_GUIDEPOST = _env_float('DARE_W_GUIDEPOST', 0.5)
W_SAFETY = _env_float('DARE_W_SAFETY', 2.0)
W_SMOOTHNESS = _env_float('DARE_W_SMOOTHNESS', 0.25)
W_COLLISION = _env_float('DARE_W_COLLISION', 3.0)

## Name and Path
run_path = f'runs/2026.06.15/13.57.54_train_diffusion_transformer_navigation_node_navigation_node'
checkpoint_name = 'epoch=0190-val_loss=0.092.ckpt'
gifs_path = f'{run_path}/inference/gifs'
