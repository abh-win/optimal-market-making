# Spec (from user, 2026-10-17)

CV pointer:
- Extended Stoikov's framework with an HMM-based micro-price indicator, classifying adverse selection via L2-L5 order book imbalances.
- Implemented a reinforcement learning agent to optimize bid-ask skew, balancing short-term price drift against strict inventory limits.
- Built a C++ tick-level limit order book backtester, simulating queue priority and realistic fill probabilities to model adverse selection costs.

New separate directory (do NOT modify ~/Projects/Market Making, which holds micro-price v1 research that may be referenced).

## Papers to anchor on
- Avellaneda & Stoikov (2008); Guéant, Lehalle & Fernandez-Tapia (2013) — closed-form quotes with inventory limits.
- Stoikov (2018) The Micro-Price (existing v1 implements it).
- Cartea, Jaimungal & Penalva (2015) — adverse selection and alpha signals in the quoting problem.
- Spooner et al. (2018) Market Making via Reinforcement Learning — RL state design, reward shaping.
- Glosten–Milgrom and Kyle — adverse-selection framing.
