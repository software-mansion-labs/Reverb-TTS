# The two-stage sampler's main path fills only these leading codebooks; one polish pass then fills the rest.
MAIN_CODEBOOKS = 2

# The main-path states of the two-stage samplers (s6-s12), fitted on progress u = k / N at the input of pass k:
# tau = s*u / (1 - (1 - s)*u), 1 - c0 = a * tau^e, 1 - c1 = b * (1 - c0)^q, with c the masked fraction of a codebook.
# Within 0.013 of every traced pass. See knowledge/inference/two_stage_sampling.md.
TRAJECTORY_SHIFT = 0.081                         # s
TRAJECTORY_C0 = (0.73, 0.80)                     # a, e
TRAJECTORY_C1 = (0.485, 1.25)                    # b, q

# Spread of a state's masked fractions around the fit, as traced: sd = base + slope * (1 - c0).
TRAJECTORY_JITTER = (0.005, 0.06)

# Share of the codebook 0-1 tokens each pass of the two-stage sN finally decides, traced from real generation
# (84 test targets x 2 seeds). Re-trace whenever the sampler changes.
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
