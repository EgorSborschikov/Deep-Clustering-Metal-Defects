import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_PATH = "Metal-surfaces-defects"
RESULTS_DIR = "metal_defects_results"
ETA_VALUES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
NUM_EPOCHS = 15
INPUT_DIM = 512
Z_DIM = 32
N_CLUSTERS = 12