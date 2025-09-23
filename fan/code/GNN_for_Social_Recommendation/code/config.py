import torch
from pathlib import Path
from utils.data_utils import load_json, load_pkl, load_sparse_matrix

# 1. 路径配置（用Path类自动适配Windows/Linux）
ROOT_PATH = Path(__file__).parent
DATA_PATH = ROOT_PATH / "data"
PROCESSED_DATA_PATH = DATA_PATH / "processed" / "ciao"  # 可切换为"epinions"
SPLITS_PATH = DATA_PATH / "splits"
RESULTS_PATH = ROOT_PATH / "results"

# 创建必要的目录
for path in [DATA_PATH / "raw", DATA_PATH / "processed", SPLITS_PATH, 
             RESULTS_PATH / "models", RESULTS_PATH / "logs", RESULTS_PATH / "figs"]:
    path.mkdir(parents=True, exist_ok=True)

# 2. 数据路径
MODEL_SAVE_PATH = RESULTS_PATH / "models" / "ciao_graphrec_best.pth"
LOG_PATH = RESULTS_PATH / "logs" / "ciao_train.log"

# 3. 数据加载（从processed文件夹加载预处理后的数据）
try:
    USER2IDX = load_json(PROCESSED_DATA_PATH / "user2idx.json")
    ITEM2IDX = load_json(PROCESSED_DATA_PATH / "item2idx.json")
    R = load_sparse_matrix(PROCESSED_DATA_PATH / "R.npz")
    T = load_sparse_matrix(PROCESSED_DATA_PATH / "T.npz")
    C = load_pkl(PROCESSED_DATA_PATH / "C.pkl")
    N = load_pkl(PROCESSED_DATA_PATH / "N.pkl")
    B = load_pkl(PROCESSED_DATA_PATH / "B.pkl")
    N_USERS = len(USER2IDX)
    N_ITEMS = len(ITEM2IDX)
except:
    # 如果预处理数据不存在，初始化默认值
    USER2IDX = {}
    ITEM2IDX = {}
    R = None
    T = None
    C = []
    N = []
    B = []
    N_USERS = 0
    N_ITEMS = 0

# 4. 模型与训练配置（匹配论文设置）
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
NUM_HIDDEN_LAYERS = 3
BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 5
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5. 汇总配置（便于外部调用）
config = {
    "root_path": ROOT_PATH,
    "n_users": N_USERS,
    "n_items": N_ITEMS,
    "n_ratings": 5,
    "embedding_dim": EMBEDDING_DIM,
    "hidden_dim": HIDDEN_DIM,
    "num_hidden_layers": NUM_HIDDEN_LAYERS,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "patience": PATIENCE,
    "lr": LR,
    "device": DEVICE,
    "R": R,
    "T": T,
    "C": C,
    "N": N,
    "B": B,
    "best_model_path": MODEL_SAVE_PATH,
    "log_path": LOG_PATH,
    "train_data_path": SPLITS_PATH / "ciao_train.csv",
    "val_data_path": SPLITS_PATH / "ciao_val.csv",
    "test_data_path": SPLITS_PATH / "ciao_test.csv",
    "processed_data_path": PROCESSED_DATA_PATH
}
