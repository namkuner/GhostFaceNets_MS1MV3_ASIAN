from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()

#rec
config.rec = "ms1-mv3-asian-face"
# Margin Base Softmax
config.margin_list = (1.0, 0.0, 0.4)
config.network = "ghostfacenets"
config.resume = True
config.checkpoint = 11
config.save_all_states = True
config.output = "ms1mv3_asian_arcface_ghostfacenets"

config.embedding_size = 512

# Partial FC
config.sample_rate = 1
config.interclass_filtering_threshold = 0

config.fp16 = False
config.batch_size =128

config.data_dir = "VILFWCut"
config.pair_path = "eval/output1.csv"

# For SGD
config.optimizer = "sgd"
config.lr = 0.008
config.momentum = 0.9
config.weight_decay = 5e-4

# For AdamW
# config.optimizer = "adamw"
# config.lr = 0.001
# config.weight_decay = 0.1

config.verbose = 2000
config.frequent = 10

# For Large Sacle Dataset, such as WebFace42M
config.dali = True
config.dali_aug = True

# Gradient ACC
config.gradient_acc = 1

# setup seed
config.seed = 2048

# dataload numworkers
config.num_workers = 1
# dataset
config.num_classes = 6579
config.num_image = 300000
config.num_epoch = 30
config.warmup_epoch = 0
config.val_targets =[]
# WandB Logger
config.wandb_key = "daa38a012f1993bc802203d31f828a53c6605938"
config.suffix_run_name = None
config.using_wandb = True
config.wandb_entity = "namkunerr"
config.wandb_project = "GhostFaceNets on Asian and MS1MV3 Dataset"
config.wandb_log_all = True
config.save_artifacts = True
config.wandb_resume = "must" # resume wandb run: Only if the you wand t resume the last run that it was interrupted
config.wandb_id = "x08r5jje"
config.wandb_resume_status =True
