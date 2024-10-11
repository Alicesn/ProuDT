sklearn_names = ["iris", "breast_cancer", "wine"]
batchSize = 512
# Define the list of seeds for 10 trials
seed_list = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
default_deep = 11
default_shallow = 8
# focal_loss
alpha = 1
gamma = 2
learning_rate = 0.01


dataset_items = [
    "Iris",  # 150 4 3
    "letter_recognition",  # 20000, 16,26
    "pen_digits",  # 10992, 16,10
    "semeion",  # 1593, 256,10
    "Segment",  # 2310 19 7
    "Splice",  # 3190 60 3
    "Rice",  # 3810 7
    "Adult",  # 32561 14
    "Bank Marketing",  # 45211 14
    "Mushroom",  # 8124 22
    "Credit Card",  # 30000 23
    "Spambase",  # 4601 57
]
config_training = {
    "preprocessing": {
        "balance_threshold1": 0.25,  # .25, #if minclass fraction less than threshold/num_classes-1 | #0=no rebalance, 1=rebalance all
        "normalization_technique": "quantile",  #'min-max'
        "quantile_noise": 1e-3,
    },
    "computation": {
        "random_seed": 42,
        "trials": 10,
        "use_best_hpo_result": True,
        "hpo_path": "_DEFAULT",
        "force_depth": False,
        "force_dropout": False,
        "force_restart": True,
        # 'use_gpu': False, #for cpu
        "use_gpu": True,  # for spartan
        "gpu_numbers": "0",  #'1',
        "n_jobs": 1,
        "verbosity": 0,
        "hpo": None,
        "search_iterations": 300,
        "cv_num": 3,
        "metrics_class": [
            "f1",
            "roc_auc",
            "accuracy",
            "total_nodes",
            "internal_node_num",
            "leaf_node_num",
        ],
        "eval_metric_class": [
            "f1",
            "roc_auc",
            "accuracy",
            "total_nodes",
            "internal_node_num",
            "leaf_node_num",
        ],
    },
}
