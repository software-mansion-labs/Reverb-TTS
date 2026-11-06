from typing import Dict, List, Literal, Tuple, Union

Row = Dict[str, Union[float, List[float]]]
Breakdown = Dict[str, Dict[str, float]]

# A traced sampler state per label: the masking rate of every codebook, and the share of the grid decided there.
Trajectory = Dict[str, Tuple[Tuple[float, ...], float]]
MultiStepMode = Literal["s6", "s8", "s10", "s12"]
