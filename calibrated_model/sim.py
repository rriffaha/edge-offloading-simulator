"""
sim.py — Step 3: the full simulator with four strategies.

    always-local : every job runs on the device (drops when the queue is full)
    always-edge  : every job is sent to the edge server
    hybrid v1    : the deadline-aware rule, Eq. (3) of the proposal
    hybrid v2    : a decision tree trained on simulated outcomes

For every job the engine first *estimates* T_L (Eq. 1) and T_E (Eq. 2) from what
it can see: the exact local queue, the average network numbers, and the edge
queueing delay Q_E last reported by the server (which is already one job old).
The job's *real* time then differs: latency has jitter, Q_E has moved on, and
the packet can be lost. That gap is what v2 is meant to learn about.

Same units as model.py (work units, ms, Mb).
"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from model import Device, Network, Job, EDGE_SERVER, FRAME_DEADLINE_MS, t_local, t_edge
from local_sim import MAX_QUEUED_JOBS


MARGIN_MS = 1.0                          # M in Eq. (3)
CONGESTION = {"low": 1.0, "high": 6.0}   # mean edge queueing delay Q_E (ms), placeholder
QE_MEMORY = 0.8                          # how strongly Q_E carries over from one job to the next
LOCAL_CAP_MS = 100.0                     # T_L stand-in when the local queue is full (can't run locally)


# ---------- Edge queueing delay over time ----------

def qe_series(n: int, mean_ms: float, rng: np.random.Generator) -> np.ndarray:
    """Q_E drifts around its mean (other players come and go), never below 0."""
    noise = rng.normal(0, 0.5 * mean_ms, n)
    qe = np.empty(n)
    prev = mean_ms
    for i in range(n):
        prev = max(0.0, mean_ms + QE_MEMORY * (prev - mean_ms) + noise[i])
        qe[i] = prev
    return qe


# ---------- What the decision engine looks at ----------

def features(job, tl, te, ql, net, qe_rep) -> list[float]:
    """Slacks (Eq. 4) plus the raw state, in a fixed order, for the tree."""
    d = job.deadline_ms
    return [d - tl, d - te, job.work, ql, net.jitter_ms, net.loss_rate, qe_rep, net.latency_ms]


# ---------- The four policies: each returns True for "offload" ----------

def policy_local(tl, te, d, f):
    return False


def policy_edge(tl, te, d, f):
    return True


def policy_v1(tl, te, d, f):
    """Eq. (3): local if safely on time, else edge if safely on time, else the faster one."""
    if tl + MARGIN_MS <= d:
        return False
    if te + MARGIN_MS <= d:
        return True
    return te < tl


def make_tree_policy(clf: DecisionTreeClassifier):
    """Walk the fitted tree by hand: far faster than clf.predict for one row at a time."""
    t = clf.tree_
    left, right = t.children_left.tolist(), t.children_right.tolist()
    feat, thr = t.feature.tolist(), t.threshold.tolist()
    leaf_class = [clf.classes_[i] for i in t.value[:, 0, :].argmax(axis=1)]

    def policy_v2(tl, te, d, f):
        node = 0
        while left[node] != -1:
            node = left[node] if f[feat[node]] <= thr[node] else right[node]
        return leaf_class[node] == 1
    return policy_v2


# ---------- Training the decision tree (v2) ----------

def train_tree(n_samples: int = 20000, seed: int = 0):
    """Random states -> simulate both choices -> label with the hindsight winner."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_samples):
        fl = rng.uniform(3, 12)
        job = Job("s", work=float(rng.lognormal(math.log(rng.uniform(30, 150)) - 0.045, 0.3)),
                  data_mb=0.2, deadline_ms=FRAME_DEADLINE_MS)
        ql = 0.0 if rng.random() < 0.4 else rng.uniform(0, 40)
        net = Network("random", latency_ms=rng.uniform(2, 50), bandwidth_mbps=rng.uniform(5, 100),
                      jitter_ms=rng.uniform(0, 12), loss_rate=rng.uniform(0, 0.06))
        net.bandwidth_down_mbps = net.bandwidth_mbps * rng.uniform(1, 3)
        qe_mean = rng.uniform(0.5, 8)
        qe_true = max(0.0, rng.normal(qe_mean, 0.5 * qe_mean))
        qe_rep = max(0.0, qe_true + rng.normal(0, 0.5 * qe_mean))  # stale/inaccurate report

        tl = ql + job.work / fl
        te_est = t_edge(job, net, EDGE_SERVER, qe_rep)
        te_real = math.inf if rng.random() < net.loss_rate else \
            t_edge(job, net, EDGE_SERVER, qe_true) + max(-net.latency_ms, rng.normal(0, net.jitter_ms))

        # The label from the proposal: local if it met the deadline, else edge if it did, else the faster.
        d = job.deadline_ms
        if tl <= d:
            label = 0
        elif te_real <= d:
            label = 1
        else:
            label = 1 if te_real < tl else 0
        X.append(features(job, min(tl, LOCAL_CAP_MS), te_est, ql, net, qe_rep))
        y.append(label)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    clf = DecisionTreeClassifier(max_depth=6, random_state=seed).fit(X_train, y_train)
    return clf, clf.score(X_test, y_test), float(np.mean(y))


# ---------- The simulation ----------

@dataclass
class Metrics:
    avg_completion_ms: float  # over jobs that finished
    miss_rate: float          # late + dropped + lost, fraction of all jobs
    drop_rate: float          # dropped (local queue full) or lost (network), fraction of all jobs
    avg_violation_ms: float   # mean of max(0, T - D) over jobs that finished
    edge_ms: float            # total edge processing time used (provider cost proxy)
    pct_offloaded: float      # fraction of jobs the engine sent to the edge
    data_sent_mb: float       # total data moved to and from the edge


def simulate(jobs: list[Job], device: Device, net: Network, qe_mean: float,
             policy, seed: int, edge: Device = EDGE_SERVER) -> Metrics:
    rng = np.random.default_rng(seed + 1000)  # network/edge luck, separate from the workload's
    n = len(jobs)
    jitter = rng.normal(0, net.jitter_ms, n)
    loss_draw = rng.random(n)
    qe = qe_series(n, qe_mean, rng)

    starts = deque()   # start times of accepted local jobs that have not started yet
    free_at = 0.0      # when the local GPU finishes everything it has accepted
    finished = []      # completion times of jobs that finished
    violations = []
    missed = dropped = offloaded = 0
    edge_ms = data_mb = 0.0

    for i, job in enumerate(jobs):
        a, d = job.arrival_ms, job.deadline_ms
        while starts and starts[0] <= a:
            starts.popleft()
        local_full = len(starts) >= MAX_QUEUED_JOBS
        ql = max(0.0, free_at - a)

        # --- the engine's estimates ---
        tl = LOCAL_CAP_MS if local_full else t_local(job, device, ql)
        qe_rep = qe[i - 1] if i else qe_mean
        te = t_edge(job, net, edge, qe_rep)
        offload = policy(tl, te, d, features(job, tl, te, ql, net, qe_rep))

        if offload:
            offloaded += 1
            data_mb += job.data_mb
            if loss_draw[i] < net.loss_rate:            # lost on the way: counts as a miss
                missed += 1
                dropped += 1
                continue
            # jitter moves only the network delay, which can shrink to 0 but not below
            completion = t_edge(job, net, edge, qe[i]) + max(-net.latency_ms, jitter[i])
            edge_ms += job.work / edge.speed
            data_mb += job.data_out_mb
        elif local_full:                                 # nowhere to put it: dropped
            missed += 1
            dropped += 1
            continue
        else:
            start = max(a, free_at)
            completion = t_local(job, device, start - a)
            free_at = a + completion
            starts.append(start)

        finished.append(completion)
        violations.append(max(0.0, completion - d))
        missed += completion > d

    return Metrics(
        avg_completion_ms=float(np.mean(finished)) if finished else math.nan,
        miss_rate=missed / n,
        drop_rate=dropped / n,
        avg_violation_ms=float(np.mean(violations)) if violations else math.nan,
        edge_ms=edge_ms,
        pct_offloaded=offloaded / n,
        data_sent_mb=data_mb,
    )
