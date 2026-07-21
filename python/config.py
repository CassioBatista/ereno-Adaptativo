"""GeneralParameters equivalent — global configuration."""

# PEP 695: type aliases
type FeatureList = list[int]
type ClassifierRef = object  # ClassifierExtended set at runtime

# Dataset path (set before running)
DATASET: str = ""

# GRASP / evaluation settings
FOLDS: int = 5
GRASP_SEED: int = 5
EVALUATION_SEED: int = 7
NUM_CLASSES: int = 2
# Penalidade de cardinalidade no criterio de selecao GRASP: a busca otimiza
# (metrica - FEATURE_PENALTY * n_features). 0.0 = comportamento original (F1 puro);
# >0 torna o GRASP mais parcimonioso (feature so entra se agrega > penalidade).
FEATURE_PENALTY: float = 0.0
NORMALIZE: bool = False
DEBUG_MODE: bool = False
CSV: bool = True
CROSS_VALIDATION: bool = True
SINGLE_FOLD_MODE: bool = False
PRINT_SELECTION: bool = False
PRINT_TRAINING_TIME: bool = False
PRINT_TESTING_TIME: bool = False

# GRASP algorithm names
GRASP_METHOD = ["GR-G-BF", "GR-G-VND", "GR-G-RVND", "F-G-VND", "F-G-RVND", "I-G-VND"]

# Runtime state (mutated during evaluation)
FEATURE_SELECTION: FeatureList = []
SINGLE_CLASSIFIER_MODE: ClassifierRef = None
CLASSIFIERS_FOREACH: list | None = None

# Train/test split (used when CROSS_VALIDATION=False)
TRAIN: tuple | None = None
TEST: tuple | None = None
