MODELS_FOLDER = "models"
DATASETS_FOLDER = "datasets"
MODEL_NAME = "decision-tree-model"

COLUMNS_TO_DROP = ['RowNumber', 'CustomerId', 'Surname']
BINARY_FEATURES = [
    "Gender",
]
ONE_HOT_ENCODE_COLUMNS = [
    "Geography",
]
MODEL_PARAMS = {
    "min_samples_split": 3,
    "min_samples_leaf": 1,
    "random_state": 8888,
}