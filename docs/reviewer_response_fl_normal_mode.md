# Reviewer response — why FL is the *normal* operating mode

**Reviewer concern.** The definition of federated learning (FL) as the normal mode of
operation is not sufficiently justified.

## The argument (single efficiency axis + the hub as discriminator)

We agree that the naive justifications are weak, and we make the honest case instead.

**Neither "fewer messages per node" nor "faster convergence" is, on its own, a valid
reason.** Both reduce to a *single* efficiency quantity. To make one node's new
knowledge (e.g., a freshly trained attack specialist) available network-wide, the
total dissemination work is `Θ(N)` in **both** paradigms — at minimum every one of the
`N` peers must be touched. That work has two faces that trade against each other:

- **communication / link** — the per-node fan-out (its *degree*), and
- **dissipation / latency** — the number of rounds for the knowledge to spread
  (the overlay *depth/diameter*).

A gossip overlay slides along this one axis: lowering the degree to 1 makes each round
as cheap **per node** as an FL client (send to one peer, receive from one), but the
diameter grows, so dissemination takes more rounds; raising the degree buys speed at a
higher link cost. **A decentralized overlay cannot minimize both at once** — for a
sparse graph the diameter is `Θ(N)` (ring/chain), and even a dense expander is bounded
below by `Θ(log N)`.

**What escapes the trade-off is a hub.** FL concentrates the whole `Θ(N)` cost on one
node — the server — which aggregates once and redistributes once. This is the *only*
configuration that is simultaneously **depth-1** (every peer receives new knowledge in
a single round, for any `N`) **and** `O(1)` per client. No decentralized scheme can
match both, because none has a node that absorbs the `Θ(N)`.

|            | total work | how it is spent      | depth (latency) | per-node cost |
|------------|-----------|----------------------|-----------------|---------------|
| **FL**     | Θ(N)      | concentrated at the hub | **1**        | O(1) / client |
| **GL** (degree `d`) | Θ(N) | distributed        | ~N/d            | O(d) / node   |

Therefore FL is the normal mode **not because it is cheaper** — the total work is the
same `Θ(N)` — but because, **while a reliable, trusted hub is available, it dominates
on every axis at once** by concentrating that cost centrally. And that same hub is
exactly the single point of failure and of *trust* that motivates the gossip fallback:
when the server is lost, partitioned, or cannot be trusted (a Byzantine aggregator),
ReSIDS switches to gossip (GL), which gives up the hub and therefore returns to the
single efficiency trade-off, paying dissemination depth `~N` as the price of
resilience. The contribution of ReSIDS is not "FL is more efficient"; it is
**surviving the loss of the hub** by switching modes at runtime.

## Ready-to-use paragraph (for the paper / response letter)

> We adopt FL as the normal operating mode not on the grounds of raw efficiency —
> disseminating one peer's update to the whole federation costs `Θ(N)` communication
> work in either paradigm, and a gossip overlay can be made as link-efficient as FL by
> reducing its degree, at the cost of proportionally slower dissipation (communication
> and convergence latency are two faces of a single trade-off). The decisive factor is
> the *hub*: by concentrating the aggregation at a single server, FL is the only
> configuration that is simultaneously depth-1 in dissemination (new knowledge reaches
> every peer in one round, independently of `N`) and `O(1)` per client — an advantage no
> decentralized overlay can match, since none has a node absorbing the `Θ(N)` cost.
> That same hub, however, is a single point of failure and of trust. ReSIDS therefore
> uses FL while a reliable, trusted server is available and switches to gossip when it
> is not, accepting the decentralized dissemination penalty (depth `~N`, Fig. X) as the
> price of resilience. The empirical cost of running gossip as the *default* is a
> new-attack detection-latency window that scales with `N` (Fig. X): at `N=100` a novel
> attack observed at a single peer needs ~100 rounds to be detected network-wide, while
> the FL hub closes that window in ~1 round for any `N`.

## Supporting figures
- `results/new_attack_scaling_latency.{pdf}` — dissemination depth: FL `O(1)`, GL `~N`.
- `results/new_attack_scaling_ramps.{pdf}` — the detection window widens with `N`.
- `results/new_attack_convergence_{1src,2src}.{pdf}` — the per-round view, and the
  corroboration (`k>=2`) result: a single-source novel attack is detectable only at
  `k>=1` until a *second* source appears (RQ4 shown dynamically).
