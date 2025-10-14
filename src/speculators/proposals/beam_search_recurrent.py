"""
Beam search recurrent token proposal method for recurrent drafting.

This implementation is based on Apple's recurrent drafting technique and contains
concepts derived from the ml-recurrent-drafter repository:
https://github.com/apple/ml-recurrent-drafter

Original work:
Copyright (C) 2024 Apple Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0

This module implements the sophisticated beam search approach used in recurrent
drafting, where multiple candidate sequences are generated simultaneously and
then verified by the base model. The implementation has been adapted for the
speculators framework with enhanced configuration and documentation.

Classes:
    BeamSearchRecurrentProposalConfig: Configuration for beam search recurrent proposal

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from speculators.config import TokenProposalConfig

__all__ = ["BeamSearchRecurrentProposalConfig"]


@TokenProposalConfig.register("beam_search_recurrent")
class BeamSearchRecurrentProposalConfig(TokenProposalConfig):
    """
    Configuration for beam search recurrent token proposal method.
    
    This method generates multiple candidate token sequences using beam search
    with RNN state management. Each beam maintains its own RNN state and explores
    different token continuations. The base model then verifies all candidates
    and selects the best sequence.
    
    This is more sophisticated than simple greedy as it:
    1. Explores multiple paths simultaneously (beam_width)
    2. Maintains longer sequences (beam_length) 
    3. Uses RNN state for better context modeling
    4. Employs sophisticated verification logic
    """
    
    proposal_type: Literal["beam_search_recurrent"] = "beam_search_recurrent"
    beam_width: int = Field(
        default=10,
        description=(
            "Number of candidate sequences (beams) to maintain during search. "
            "Higher values explore more possibilities but increase computation. "
            "Typical values range from 5-20."
        ),
        ge=1,
    )
    beam_length: int = Field(
        default=4,
        description=(
            "Maximum length of each candidate sequence. This determines how many "
            "tokens ahead the speculator looks. Higher values can improve "
            "acceptance rates but increase computation."
        ),
        ge=2,  # Must be at least 2 for meaningful beam search
    )
    top_k: int = Field(
        default=50,
        description=(
            "Top-k filtering applied to logits before beam search. Restricts "
            "the vocabulary to the k most likely tokens at each step."
        ),
        ge=1,
    )
    use_rnn_state: bool = Field(
        default=True,
        description=(
            "Whether to use RNN state updates during beam search. If False, "
            "uses simple addition of embeddings to state."
        ),
    )
    greedy_verification: bool = Field(
        default=False,
        description=(
            "Whether to use greedy verification (exact token matching) or "
            "probabilistic verification (rejection sampling). Greedy is faster "
            "but may be less effective."
        ),
    )
