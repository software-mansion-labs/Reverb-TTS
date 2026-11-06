# ---------- Optimizer ----------

# Momentum of AdamW. The second one is lower than torch's default, which is what
# language model training settled on: it follows the curvature more closely.
BETAS = (0.9, 0.95)
