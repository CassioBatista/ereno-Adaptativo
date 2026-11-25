"""Load and partition a dataset into N client splits.

Supported partitioners (mirrors flwr-datasets strategies, pure numpy):

  iid          — equal-size random splits, IID distribution (default)
  dirichlet    — class-heterogeneous splits via Dirichlet(alpha)
                 alpha << 1 → very heterogeneous; alpha >> 1 → near-IID
  shard        — each client receives shards_per_client contiguous class shards
  exponential  — exponentially unequal volumes (client 0 largest)
  linear       — linearly unequal volumes (client 0 largest)
  attack       — each client receives ONE attack class + an IID slice of the
                 normal class (specialist sensors; use num_clients == number
                 of attack classes for a strict 1:1 mapping)
  specialists  — evaluation-v1 (Paper 2): one specialist per attack type (types under
                 100 rows share one), placed on the nodes in contiguous blocks by size;
                 each node trains one binary model per specialist it holds against its
                 slice of the benign rows (pass groups_out=[] to receive the groups)
"""

import numpy as np
import python.util as util


# ── public API ────────────────────────────────────────────────────────────────

def load_and_partition(
    filepath: str,
    feature_indices: list[int],
    num_clients: int,
    seed: int = 42,
    partitioner: str = "iid",
    **kwargs,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Load ARFF, apply feature selection, return per-client (X, y) partitions.

    Parameters
    ----------
    filepath        : path to the CSV/ARFF dataset
    feature_indices : 1-based feature indices selected by GRASP
    num_clients     : number of federation clients
    seed            : RNG seed for reproducibility
    partitioner     : one of 'iid', 'dirichlet', 'shard', 'exponential', 'linear'
    **kwargs        : extra args forwarded to the chosen partitioner:
                      dirichlet  → alpha (float, default 0.5)
                      shard      → shards_per_client (int, default 2)
                      exponential / linear → (no extra args)

    Returns
    -------
    (partitions, X_filtered, y_all)
      partitions  : list of (X, y) arrays, one per client
      X_filtered  : full dataset with only the selected features
      y_all       : full label array
    """
    X_all, y_all, _ = util.load_arff(filepath)

    cols = [i - 1 for i in sorted(feature_indices)]
    Xf   = X_all[:, cols]

    rng = np.random.default_rng(seed)

    # benign_cap: limita a classe normal a N amostras (subamostra aleatória),
    # mantendo TODOS os ataques. Controle de memória para datasets muito
    # desbalanceados e redundantes (ex.: ERENO, 93% normal) — os ataques,
    # que são o sinal escasso, ficam intactos.
    benign_cap = kwargs.get("benign_cap")
    if benign_cap:
        benign_cap = int(benign_cap)
        normal = util.normal_class
        normal_idx = np.where(y_all == normal)[0]
        if len(normal_idx) > benign_cap:
            drop = rng.permutation(normal_idx)[benign_cap:]
            keep = np.ones(len(y_all), dtype=bool)
            keep[drop] = False
            Xf, y_all = Xf[keep], y_all[keep]
            print(f"[benign_cap] classe normal limitada a {benign_cap:,} "
                  f"(de {len(normal_idx):,}); ataques preservados")

    match partitioner:
        case "specialists":
            groups = _specialist_groups(y_all, num_clients, rng)
            splits = [g["rows"] for g in groups]
            if kwargs.get("groups_out") is not None:
                kwargs["groups_out"].extend(
                    [{"specialists": g["specialists"], "names": g["names"]} for g in groups])
        case "attack":
            splits = _attack_per_client(Xf, y_all, num_clients, rng)
        case "iid":
            splits = _iid(Xf, y_all, num_clients, rng)
        case "dirichlet":
            alpha = float(kwargs.get("alpha", 0.5))
            splits = _dirichlet(Xf, y_all, num_clients, alpha, rng)
        case "shard":
            shards_per_client = int(kwargs.get("shards_per_client", 2))
            splits = _shard(Xf, y_all, num_clients, shards_per_client, rng)
        case "exponential":
            splits = _unequal(Xf, y_all, num_clients, "exponential", rng)
        case "linear":
            splits = _unequal(Xf, y_all, num_clients, "linear", rng)
        case _:
            raise ValueError(
                f"Particionador inválido: '{partitioner}'. "
                f"Opções: iid, dirichlet, shard, exponential, linear, attack."
            )

    partitions = [(Xf[idx], y_all[idx]) for idx in splits]
    return partitions, Xf, y_all


# ── partitioners ──────────────────────────────────────────────────────────────

SMALL = 100   # types with fewer training rows share one specialist (evaluation-v1)


def _specialist_units(counts: list[int], normal: int, num_clients: int):
    """Specialists of evaluation-v1 and their placement (docs/EVALUATION_PROTOCOL.md,
    section 2.3; scripts/ev1_partition_n14.py with N = num_clients):
      one specialist per attack type; the types with fewer than SMALL rows share one;
      specialists in decreasing order of rows (ties: class declaration order);
      fewer specialists than nodes -> the extra nodes go to the largest specialists, one
      at a time, a specialist on k nodes having its rows split into k parts;
      otherwise contiguous blocks: q = S div N per node, the last r = S mod N nodes q + 1.
    Returns [(codes, rows, node)] in placement order and whether specialists were split."""
    types = [c for c in range(len(counts)) if c != normal and counts[c] > 0]
    big = [c for c in types if counts[c] >= SMALL]
    small = [c for c in types if counts[c] < SMALL]
    sp = [([c], counts[c]) for c in big]
    if small:
        sp.append((small, sum(counts[c] for c in small)))
    sp.sort(key=lambda s: (-s[1], min(s[0])))
    if len(sp) < num_clients:
        copies = [1] * len(sp)
        for i in range(num_clients - len(sp)):
            copies[i % len(sp)] += 1
        units, node = [], 0
        for (cs, rows), k in zip(sp, copies):
            for part in np.array_split(np.arange(rows), k):
                units.append((cs, len(part), node))
                node += 1
        return units, True
    q, r = divmod(len(sp), num_clients)
    node_of = [i for i, k in enumerate([q] * (num_clients - r) + [q + 1] * r) for _ in range(k)]
    return [(cs, rows, node_of[j]) for j, (cs, rows) in enumerate(sp)], False


def _specialist_groups(y: np.ndarray, num_clients: int, rng: np.random.Generator):
    """Partitioner 'specialists' (evaluation-v1): node i holds the i-th of N disjoint
    random slices of the benign rows plus the attack rows of the specialists placed on it,
    and trains one binary model per specialist (fd/client_xgb.py). Same random draws as
    the 'attack' partitioner (benign shuffle, then the per-class shuffles when specialists
    are split over nodes), hence the same partitions as scripts/ev1_step1.py.
    Returns, per node: rows (indices into y), specialists [[codes], ...], names."""
    normal = util.normal_class
    benign = np.where(y == normal)[0].copy()
    rng.shuffle(benign)
    slices = np.array_split(benign, num_clients)
    counts = np.bincount(y, minlength=int(y.max()) + 1).tolist()
    units, split = _specialist_units(counts, normal, num_clients)
    nodes = [{"att": [], "specialists": []} for _ in range(num_clients)]
    if split:
        types = sorted((int(c) for c in np.unique(y) if c != normal),
                       key=lambda c: -int((y == c).sum()))
        per = {c: sum(1 for u in units if u[0] == [c]) for c in types}
        cid = 0
        for c in types:
            idx = np.where(y == c)[0].copy()
            rng.shuffle(idx)
            for part in np.array_split(idx, per[c]):
                cs, rows, node = units[cid]
                assert cs == [c] and node == cid and rows == len(part), (cs, c, cid)
                nodes[node]["att"].append(np.sort(part))
                nodes[node]["specialists"].append([c])
                cid += 1
    else:
        for cs, rows, node in units:
            idx = np.where(np.isin(y, cs))[0]
            assert len(idx) == rows
            nodes[node]["att"].append(idx)
            nodes[node]["specialists"].append(list(cs))
    out = []
    for i, nd in enumerate(nodes):
        rows = np.concatenate(nd["att"] + [slices[i]]) if nd["att"] else slices[i]
        out.append({"rows": rows, "specialists": nd["specialists"],
                    "names": [",".join(map(str, cs)) for cs in nd["specialists"]]})
    print(f"[specialists] {sum(len(n['specialists']) for n in nodes)} specialists on "
          f"{num_clients} nodes: " + " ".join(str(len(n["specialists"])) for n in nodes))
    return out

def _attack_per_client(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Each client receives ONE attack class + an IID slice of the normal class.

    Models a network of specialist sensors: every client knows the benign
    baseline but has only ever seen a single attack type. Attack classes are
    assigned in descending-size order.

    num_clients vs number of attack classes:
      ==  strict 1:1 mapping (one specialist per attack);
      >   redundant sensors: extra clients are assigned to the LARGEST
          classes, one at a time, splitting that class's samples into IID
          slices (e.g. ERENO with 10 clients: the three 39k classes get two
          specialists each, the other four get one);
      <   leftover classes are distributed round-robin (warning printed).
    """
    normal = util.normal_class

    benign_idx = np.where(y == normal)[0].copy()
    rng.shuffle(benign_idx)
    benign_slices = np.array_split(benign_idx, num_clients)

    attack_classes = sorted(
        (int(c) for c in np.unique(y) if c != normal),
        key=lambda c: -int((y == c).sum()),
    )
    n_att = len(attack_classes)

    client_indices: list[list[int]] = [s.tolist() for s in benign_slices]
    assignment: list[list[int]] = [[] for _ in range(num_clients)]

    if num_clients > n_att:
        # sensores redundantes: excedentes vão às maiores classes, 1 por vez
        sensors_per_class = {c: 1 for c in attack_classes}
        for i in range(num_clients - n_att):
            sensors_per_class[attack_classes[i % n_att]] += 1
        dup = {c: k for c, k in sensors_per_class.items() if k > 1}
        print(f"[attack] {num_clients} clientes para {n_att} classes — "
              f"sensores redundantes (classe: nº de sensores): {dup}")

        cid = 0
        for cls in attack_classes:
            cls_idx = np.where(y == cls)[0].copy()
            rng.shuffle(cls_idx)
            for part in np.array_split(cls_idx, sensors_per_class[cls]):
                client_indices[cid].extend(part.tolist())
                assignment[cid].append(cls)
                cid += 1
    else:
        if n_att > num_clients:
            print(f"[attack] AVISO: {n_att} classes de ataque para "
                  f"{num_clients} clientes — excedentes vão em round-robin "
                  f"(use num_clients={n_att} para 1 ataque/cliente).")
        for j, cls in enumerate(attack_classes):
            cid = j % num_clients
            client_indices[cid].extend(np.where(y == cls)[0].tolist())
            assignment[cid].append(cls)

    for cid in range(num_clients):
        n_benign = len(benign_slices[cid])
        n_total  = len(client_indices[cid])
        print(f"[attack] cliente {cid}: classes de ataque {assignment[cid]} "
              f"({n_total - n_benign} amostras) + {n_benign} benignas")

    return [rng.permutation(np.array(idx, dtype=np.intp))
            for idx in client_indices]


def _iid(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Equal-size random splits — IID distribution."""
    indices = rng.permutation(len(X))
    return np.array_split(indices, num_clients)


def _dirichlet(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Class-heterogeneous splits via Dirichlet(alpha).

    alpha << 1  →  each client sees very few classes (high heterogeneity)
    alpha >> 1  →  near-IID distribution across clients
    """
    classes = np.unique(y)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)

        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        # ensure no client gets 0 samples from this class
        proportions = np.maximum(proportions, 1e-6)
        proportions /= proportions.sum()

        boundaries = (np.cumsum(proportions) * len(cls_indices)).astype(int)
        boundaries = np.clip(boundaries, 0, len(cls_indices))

        splits = np.split(cls_indices, boundaries[:-1])
        for cid, part in enumerate(splits):
            client_indices[cid].extend(part.tolist())

    return [np.array(idx, dtype=np.intp) for idx in client_indices]


def _shard(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    shards_per_client: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Each client receives shards_per_client contiguous class-sorted shards.

    Samples are sorted by label, then split into num_clients * shards_per_client
    shards. Each client receives shards_per_client randomly assigned shards.
    """
    total_shards = num_clients * shards_per_client
    sorted_indices = np.argsort(y, stable=True)
    shards = np.array_split(sorted_indices, total_shards)

    shard_ids = rng.permutation(total_shards)
    client_indices = []
    for cid in range(num_clients):
        assigned = shard_ids[cid * shards_per_client:(cid + 1) * shards_per_client]
        client_indices.append(np.concatenate([shards[s] for s in assigned]))

    return client_indices


def _unequal(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    mode: str,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Volume-heterogeneous splits with equal class distribution per client.

    mode='exponential' : weights ∝ 2^i  (client 0 gets most data)
    mode='linear'      : weights ∝ (N-i) (client 0 gets most data)
    """
    i = np.arange(num_clients, dtype=float)
    if mode == "exponential":
        weights = 2.0 ** i
    else:  # linear
        weights = (num_clients - i)

    weights /= weights.sum()
    sizes = np.round(weights * len(X)).astype(int)
    # fix rounding so sizes sum exactly to len(X)
    diff = len(X) - sizes.sum()
    sizes[0] += diff

    indices = rng.permutation(len(X))
    splits = []
    start = 0
    for size in sizes:
        splits.append(indices[start:start + size])
        start += size

    return splits
