MIN_MODEL_ACCURACY = 0.95

DATASET_PATH = r"data\20200318-174041"
# DATASET_PATH = r"data\min"
DATASET_FOLDERS = ("real", "digital fake", "printed fake")

LABELS_ORDER = ['digital fake', 'printed fake', 'real']

SVM_C = 200
SVM_KERNEL = 'rbf'
SVM_LBP_TYPE = "histogram"
