"""GraspSolution equivalent."""

import copy

import python.config as config


class GraspSolution:
    """Holds the current feature selection state: selected features + RCL pool."""

    def __init__(
        self,
        solution_features: list[int] | None = None,
        rcl_features: list[int] | None = None,
    ):
        self._solution: list[int] = list(solution_features) if solution_features else []
        self._rcl: list[int]      = list(rcl_features)      if rcl_features      else []
        self.evaluation = None  # Result
        self.current_time_seconds: float = 0.0

    # ------------------------------------------------------------------
    # Feature management
    # ------------------------------------------------------------------
    def add_feature(self, f: int)      -> None: self._solution.append(f)
    def add_feature_flip(self, f: int) -> None: self._rcl.append(f)
    def add_feature_rcl(self, f: int)  -> None: self._rcl.append(f)

    def deselect_feature(self, index: int) -> None:
        self._rcl.append(self._solution.pop(index))

    def select_feature(self, rcl_index: int) -> None:
        self._solution.append(self._rcl.pop(rcl_index))

    def replace_feature(self, sol_index: int, rcl_index: int) -> None:
        self._solution.append(self._rcl.pop(rcl_index))
        self._rcl.append(self._solution.pop(sol_index))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_array_features(self)         -> list[int]: return list(self._solution)
    def get_num_selected_features(self)  -> int:       return len(self._solution)
    def get_num_rcl_features(self)       -> int:       return len(self._rcl)
    def get_selected_features(self)      -> list[int]: return self._solution
    def get_rcl_features(self)           -> list[int]: return self._rcl
    def copy_solution_features(self)     -> list[int]: return list(self._solution)
    def copy_rcl_features(self)          -> list[int]: return list(self._rcl)
    def get_feature_set(self)            -> str:       return str(self._solution)

    def get_accuracy(self) -> str:
        return str(self.evaluation.accuracy) if self.evaluation else "N/A"

    def get_f1_score(self) -> str:
        return str(self.evaluation.f1score) if self.evaluation else "N/A"

    def get_features_and_performance(self) -> str:
        f1 = self.evaluation.f1score if self.evaluation else 0.0
        return f"{self._solution} - F1Score: {f1}"

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    def is_better_than(self, other: "GraspSolution", metric: str = "F1SCORE") -> bool:
        if other.get_num_selected_features() == 0: return True
        if self.get_num_selected_features()  == 0: return False

        match metric:
            case "ACCURACY":
                s = self.evaluation.accuracy  if self.evaluation  else 0.0
                o = other.evaluation.accuracy if other.evaluation else 0.0
            case _:  # F1SCORE
                s = self.evaluation.f1score   if self.evaluation  else 0.0
                o = other.evaluation.f1score  if other.evaluation else 0.0

        # Penalidade de cardinalidade: a busca otimiza (metrica - lambda*n_features),
        # tornando o GRASP parcimonioso. FEATURE_PENALTY=0 preserva o comportamento
        # original (F1 puro, com desempate por tamanho abaixo).
        lam = getattr(config, "FEATURE_PENALTY", 0.0)
        if lam:
            s -= lam * self.get_num_selected_features()
            o -= lam * other.get_num_selected_features()

        if s == o:
            return self.get_num_selected_features() < other.get_num_selected_features()
        return s > o

    # ------------------------------------------------------------------
    # Cloning
    # ------------------------------------------------------------------
    def new_clone(self, reset_metrics: bool) -> "GraspSolution":
        c = GraspSolution(list(self._solution), list(self._rcl))
        if not reset_metrics and self.evaluation is not None:
            c.evaluation = copy.copy(self.evaluation)
        c.current_time_seconds = self.current_time_seconds
        return c
