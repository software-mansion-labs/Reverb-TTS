import torch

# ---------- Audio codec ----------

# Frames the audio codec produces per second of audio. Used to turn durations,
# such as the minimum prefix length, into a number of frames.
FRAME_RATE = 25

# The id standing for a masked audio token. It is the last entry of the codec's
# vocabulary, and doubles as the padding value, since padded positions are
# excluded from attention anyway.
AUDIO_MASK_ID = 1024


# ---------- Labels ----------

# The label standing for "no loss here". 
# Positions carrying it are skipped by the cross entropy.
IGNORE_INDEX = -100


# ---------- Prompting ----------

# The tokenizer the pretrained model was trained with.
DEFAULT_TOKENIZER = "k2-fsa/OmniVoice"

# The prompt the model is conditioned on. Instructions are not used yet, so the
# slot is filled with the same "None" the pretrained model was trained with.
PROMPT_TEMPLATE = "<|lang_start|>{language}<|lang_end|><|instruct_start|>None<|instruct_end|>"

# How a transcript is wrapped before it is tokenized.
TEXT_TEMPLATE = "<|text_start|>{text}<|text_end|>"

# Stands in for the prompt and the text of a sample whose prefix was dropped,
# keeping every sample the same shape of dict.
EMPTY = torch.empty(0, dtype=torch.long)


# ---------- Batching ----------

# How many batches are filled at the same time. Samples join the first one they
# fit into, so a few open batches are enough to keep every length served.
OPEN_BATCHES = 8
