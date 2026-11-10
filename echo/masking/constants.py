# The two-stage sampler's main path fills only these leading codebooks; one polish pass then fills the rest.
MAIN_CODEBOOKS = 2

# The main-path states of the deployed samplers, traced on the RL round-5 model.
MAIN_PATH_SAMPLERS = {
    "s8": {
        "masked": ((1.0000, 1.0000), (0.9755, 0.9915), (0.9457, 0.9811), (0.9065, 0.9671), (0.8537, 0.9471),
                   (0.7758, 0.9167), (0.6500, 0.8623), (0.4111, 0.7355)),
        "spread": ((0.0000, 0.0000), (0.0082, 0.0078), (0.0118, 0.0112), (0.0155, 0.0142), (0.0200, 0.0176),
                   (0.0242, 0.0216), (0.0279, 0.0264), (0.0324, 0.0319)),
        "decisions": (0.0159, 0.0199, 0.0260, 0.0359, 0.0536, 0.0896, 0.1825, 0.5767),
    },
    "s8_rmc025": {
        "masked": ((1.0000, 1.0000), (0.9755, 0.9915), (0.9516, 0.9845), (0.9205, 0.9770), (0.8770, 0.9655),
                   (0.8127, 0.9492), (0.7070, 0.9192), (0.5087, 0.8428)),
        "spread": ((0.0000, 0.0000), (0.0082, 0.0078), (0.0118, 0.0106), (0.0142, 0.0126), (0.0179, 0.0156),
                   (0.0212, 0.0189), (0.0243, 0.0233), (0.0298, 0.0290)),
        "decisions": (0.0066, 0.0098, 0.0142, 0.0208, 0.0325, 0.0550, 0.1825, 0.6786),
    },
    "s6_rmc025": {
        "masked": ((1.0000, 1.0000), (0.9678, 0.9885), (0.9332, 0.9787), (0.8794, 0.9643), (0.7887, 0.9388),
                   (0.6025, 0.8723)),
        "spread": ((0.0000, 0.0000), (0.0091, 0.0091), (0.0121, 0.0117), (0.0154, 0.0147), (0.0206, 0.0196),
                   (0.0276, 0.0265)),
        "decisions": (0.0086, 0.0147, 0.0247, 0.0452, 0.1678, 0.7390),
    },
}

# The least probability a pass of a sampler is drawn with: every pass keeps at least this share, the rest follow
# the decision shares (only the early passes, which decide little, are raised).
DEFAULT_MIN_PASS_SHARE = 0.01

# ---------- The v0.0.2 recipe's schedule (masking.stage_shares) ----------

# The main-path states of the two-stage samplers (s6-s12) before RL (remask 0.5), fitted on progress u = k / N at the input of pass k:
# tau = s*u / (1 - (1 - s)*u), 1 - c0 = a * tau^e, 1 - c1 = b * (1 - c0)^q, with c the masked fraction of a codebook.
# Within 0.013 of every traced pass. See knowledge/inference/two_stage_sampling.md.
TRAJECTORY_SHIFT = 0.081                         # s
TRAJECTORY_C0 = (0.73, 0.80)                     # a, e
TRAJECTORY_C1 = (0.485, 1.25)                    # b, q

# Spread of a state's masked fractions around the fit, as traced: sd = base + slope * (1 - c0).
TRAJECTORY_JITTER = (0.005, 0.06)

# Share of the codebook 0-1 tokens each pass of the two-stage sN (remask 0.5) finally decides, traced from real
# generation (84 test targets x 2 seeds).
DECISION_SHARES = {
    6:  (0.002, 0.002, 0.003, 0.006, 0.168, 0.819),
    8:  (0.002, 0.002, 0.003, 0.005, 0.008, 0.017, 0.183, 0.780),
    10: (0.002, 0.002, 0.003, 0.004, 0.005, 0.008, 0.012, 0.025, 0.190, 0.749),
    12: (0.001, 0.002, 0.002, 0.003, 0.004, 0.006, 0.007, 0.010, 0.016, 0.033, 0.192, 0.724),
}

# Passes counted as early when the stages are weighted explicitly.
EARLY_PASSES = 3

# ---------- Training defaults ----------

# Share of samples drawn at the polish state.
DEFAULT_POLISH_SHARE = 0.25

# Loss weights of codebooks 2-7 within a polish sample.
DEFAULT_POLISH_WEIGHTS = (8.0, 8.0, 4.0, 4.0, 2.0, 2.0)

# How far the tail reaches past the last main pass: the least masked fraction of codebook 0.
DEFAULT_TAIL_MASKED = 0.40
