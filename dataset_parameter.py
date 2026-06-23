## Dataset Options
DATASET_METHOD = 'navigation_astar'  # A* start-to-goal expert demonstrations.
DATA_TYPE = 'node'  # 'map', 'node'
USE_TEST_DATASET = False  # False for train dataset, True for test dataset
USE_DELTA_POSITION = True # False for absolute position, True for delta position
TEST_N_AGENTS = 1 # SINGLE AGENT keep it 1
EXPERT_PLANNER_MAP = 'ground_truth_oracle'  # 'ground_truth_oracle' or 'belief'

## Environment Runner Options
USE_GPU = False
DEVICE = 'cpu'
NUM_META_AGENT = 24  # the number of processes

NUM_EPISODES = 10000  # Reuse maps with a new randomized start/goal per episode.
NUM_RUN = 1
SAVE_GIFS = True  # Save one rollout GIF for every collected episode.

## Name and Path
DESCRIPTION = ''
dataset_path = f'dataset/{DATASET_METHOD}_{DATA_TYPE}_{"test" if USE_TEST_DATASET else "train"}_{NUM_EPISODES}{"_" + DESCRIPTION if DESCRIPTION else ""}'
gifs_path = f'{dataset_path}/gifs'
