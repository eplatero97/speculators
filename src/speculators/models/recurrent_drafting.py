"""
Speculators implementation of Recurrent Drafting.

This implementation is based on Apple's recurrent drafting technique and contains
code derived from the ml-recurrent-drafter repository:
https://github.com/apple/ml-recurrent-drafter

Original work:
Copyright (C) 2024 Apple Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0

Modifications for speculators framework integration:
- Adapted to speculators framework architecture and patterns
- Added configuration classes and registration decorators
- Integrated with speculators proposal methods and verifier system
- Enhanced error handling and generation parameters
- Added comprehensive documentation and type hints

Classes:
    RecurrentDraftingConfig: Configuration class for recurrent drafting speculator
    RecurrentDraftingSpeculator: Main model implementation for recurrent drafting
    ResBlock: Residual block used in the drafter head

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

import os
from typing import Any, ClassVar, Literal, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from pydantic import Field
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from speculators import SpeculatorModel, SpeculatorModelConfig
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.proposals.beam_search_recurrent import BeamSearchRecurrentProposalConfig

__all__ = [
    "RecurrentDraftingConfig",
    "RecurrentDraftingSpeculator",
    "ResBlock",
    "BeamShape",
]

# Constants from original implementation
LOG_0 = -50000.0
LOG_1 = 0.0


@dataclass
class BeamShape:
    """Configuration for beam search dimensions."""
    width: int  # Number of beams to maintain
    length: int  # Length of each beam sequence


@SpeculatorModelConfig.register("recurrent_drafting")
class RecurrentDraftingConfig(SpeculatorModelConfig):
    """
    Configuration for Recurrent Drafting speculator.
    
    Recurrent drafting uses RNN state management combined with beam search
    to generate multiple candidate token sequences that are then verified
    by the base model.
    
    :param hidden_size: Hidden size matching the base model
    :param vocab_size: Vocabulary size of the base model
    :param exit_dim: Dimension of the drafter's internal representations
    :param num_draft_layers: Number of residual blocks in the drafter head
    :param rnn: Whether to use RNN state updates (vs simple addition)
    :param emb_norm: Whether to include embedding normalization
    :param rms_norm_eps: Epsilon for RMS normalization
    """
    
    speculators_model_type: Literal["recurrent_drafting"] = "recurrent_drafting"
    architectures: list[str] = Field(
        default_factory=lambda: ["RecurrentDraftingSpeculator"],
        description="Model architectures that can load these weights",
    )
    hidden_size: int = Field(
        description="Hidden size matching the base model"
    )
    vocab_size: int = Field(
        description="Vocabulary size of the base model"
    )
    exit_dim: int = Field(
        description="Dimension of the drafter's internal representations"
    )
    num_draft_layers: int = Field(
        description="Number of residual blocks in the drafter head"
    )
    rnn: bool = Field(
        default=False,
        description="Whether to use RNN state updates (vs simple addition)"
    )
    emb_norm: bool = Field(
        default=False,
        description="Whether to include embedding normalization"
    )
    rms_norm_eps: float = Field(
        default=1e-5,
        description="Epsilon for RMS normalization"
    )
    
    def __init__(self, **kwargs):
        # Set up default speculators_config if not provided
        if 'speculators_config' not in kwargs:
            kwargs['speculators_config'] = SpeculatorsConfig(
                algorithm="recurrent_drafting",
                proposal_methods=[BeamSearchRecurrentProposalConfig()],
                default_proposal_method="beam_search_recurrent",
                verifier=VerifierConfig(
                    name_or_path=None,
                    architectures=[]
                )
            )
        super().__init__(**kwargs)


class ResBlock(nn.Module):
    """Residual block used in the drafter head."""
    
    def __init__(self, config: RecurrentDraftingConfig):
        super().__init__()
        self.linear = nn.Linear(config.exit_dim, config.exit_dim, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.silu(self.linear(x))


def maintain_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Normalize logits to avoid numerical instability.
    
    Subtracts the max value from each beam to prevent all values becoming -inf.
    
    Args:
        logits: (batch_size, beam_width, vocab_size) logits from drafter
    Returns:
        logits: Normalized logits
    """
    bs, beam_width, vocab_size = logits.shape
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
    return logits - max_logits


def warp_logits(logits: torch.Tensor, top_k: int = 50, mask_value: float = LOG_0) -> torch.Tensor:
    """
    Apply top-k filtering to logits.
    
    Args:
        logits: Input logits tensor
        top_k: Number of highest probability tokens to keep
        mask_value: Value to set for filtered tokens
    Returns:
        logits: Top-k filtered logits
    """
    top_k = min(top_k, logits.shape[-1])
    # Get the k-th largest value for each position
    kth_largest = torch.topk(logits, top_k, dim=-1)[0][..., -1:]
    # Mask tokens below the k-th largest
    mask = logits < kth_largest
    return torch.where(mask, mask_value, logits)


@SpeculatorModel.register("recurrent_drafting")
class RecurrentDraftingSpeculator(SpeculatorModel):
    """
    Recurrent Drafting speculator implementation.
    
    This model uses RNN state management combined with beam search to generate
    multiple candidate token sequences. The candidates are then verified by
    the base model using sophisticated beam selection logic.
    """
    
    config_class: ClassVar[type[RecurrentDraftingConfig]] = RecurrentDraftingConfig
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [
        "verifier*",
    ]
    
    def __init__(
        self,
        config: RecurrentDraftingConfig,
        verifier: Optional[Union[str, os.PathLike, PreTrainedModel]] = None,
        verifier_attachment_mode: Optional[Literal["detached", "full", "train_only"]] = None,
        **kwargs,
    ):
        """Initialize the recurrent drafting speculator."""
        if not isinstance(config, RecurrentDraftingConfig):
            raise ValueError(f"config must be RecurrentDraftingConfig, got {type(config)}")
        
        super().__init__(
            config=config,
            verifier=verifier,
            verifier_attachment_mode=verifier_attachment_mode,
            **kwargs,
        )
        
        self.config: RecurrentDraftingConfig = config
        
        # Input projection (optional)
        input_dim = 2 * config.hidden_size
        if input_dim != config.exit_dim:
            self.input_proj = nn.Linear(input_dim, config.exit_dim, bias=True)
        
        # Drafter head with residual blocks
        layers = []
        for _ in range(config.num_draft_layers):
            layers.append(ResBlock(config))
        layers.append(nn.Linear(config.exit_dim, config.vocab_size, bias=False))
        self.lm_head = nn.Sequential(*layers)
        
        # RNN components (optional)
        if config.rnn:
            self.rnn_u = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            self.rnn_w = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # Embedding normalization (optional)
        if config.emb_norm:
            self.emb_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.post_init()
    
    def compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits from input features."""
        if hasattr(self, "input_proj"):
            x = self.input_proj(x)
        
        x = self.lm_head(x)
        x = maintain_logits(x)
        return warp_logits(x)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[torch.FloatTensor, CausalLMOutputWithPast]:
        """
        Forward pass for training.
        
        Args:
            input_ids: Input token IDs
            hidden_states: Hidden states from the verifier model
            **kwargs: Additional arguments
        Returns:
            Model outputs with logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get embeddings from verifier
        if self.verifier is None:
            raise ValueError("Verifier must be attached for forward pass")
        
        embeddings = self.verifier.get_input_embeddings()
        input_embeds = embeddings(input_ids)
        
        # Concatenate embeddings and hidden states
        combined_input = torch.cat([hidden_states, input_embeds], dim=-1)
        
        # Compute logits
        logits = self.compute_logits(combined_input)
        
        if not return_dict:
            return logits
        
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )
    
    def beam_search_candidates(
        self,
        prompt_state: torch.Tensor,
        init_token: torch.Tensor,
        embeddings: nn.Embedding,
        beam_shape: BeamShape = BeamShape(10, 4),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate candidate token sequences using beam search.
        
        Args:
            prompt_state: (batch_size, hidden_size) Hidden state from last prompt token
            init_token: (batch_size,) Initial token from base model
            embeddings: Embedding layer from the base model
            beam_shape: Beam search configuration
            
        Returns:
            beams: (batch_size, beam_width, beam_length) Candidate sequences
            logits_token_in_beam: (batch_size, beam_width, beam_length-1, vocab_size) Logits
        """
        assert prompt_state.ndim == 2
        assert init_token.ndim == 1
        assert beam_shape.length > 1
        
        batch_size = prompt_state.shape[0]
        hidden_size = prompt_state.shape[1]
        vocab_size = embeddings.num_embeddings
        device = prompt_state.device
        
        # Initialize beam probabilities
        # Only first beam starts with log(1)=0, others with log(0)=-inf
        log_p_beam = torch.full(
            (batch_size, beam_shape.width),
            LOG_0,
            device=device,
            dtype=prompt_state.dtype
        )
        log_p_beam[:, 0] = LOG_1
        
        # Replicate context for all beams
        context = prompt_state.unsqueeze(1).repeat(1, beam_shape.width, 1)
        
        # Initialize beams with init_token
        beams = init_token.unsqueeze(1).unsqueeze(2).repeat(1, beam_shape.width, 1)
        
        # Initialize RNN states
        state = torch.zeros(
            batch_size, beam_shape.width, hidden_size,
            device=device, dtype=prompt_state.dtype
        )
        
        # Store logits for each step
        logits_token_in_beam = torch.empty(
            batch_size, beam_shape.width, 0, vocab_size,
            device=device, dtype=prompt_state.dtype
        )
        
        for step in range(beam_shape.length - 1):
            # Update RNN state with previous token
            prev_token_embeds = embeddings(beams[..., -1])
            
            if self.config.rnn and hasattr(self, 'rnn_w') and hasattr(self, 'rnn_u'):
                # RNN update: state = silu(W * embed + U * state)
                state = torch.nn.functional.silu(
                    self.rnn_w(prev_token_embeds) + self.rnn_u(state)
                )
            else:
                # Simple addition
                state = prev_token_embeds + state
            
            # Compute next token logits
            combined_input = torch.cat([context, state], dim=-1)
            logits_new_token = self.compute_logits(combined_input)
            log_p_new_token = torch.log_softmax(logits_new_token, dim=-1)
            
            # Combine with beam probabilities
            log_p_beam_new_token = log_p_new_token + log_p_beam.unsqueeze(-1)
            
            # Reshape for top-k selection across all beam*vocab combinations
            flat_probs = log_p_beam_new_token.view(batch_size, -1)
            
            # Select top beam_width candidates
            top_log_probs, top_indices = torch.topk(flat_probs, beam_shape.width, dim=-1)
            
            # Convert flat indices back to (beam, token) indices
            top_beam_indices = top_indices // vocab_size
            top_token_ids = top_indices % vocab_size
            
            # Update beam probabilities
            log_p_beam = top_log_probs
            
            # Gather selected beams and extend with new tokens
            beams = self._gather_beams(beams, top_beam_indices)
            beams = torch.cat([beams, top_token_ids.unsqueeze(-1)], dim=-1)
            
            # Update states for selected beams
            state = self._gather_beams(state, top_beam_indices)
            
            # Store logits for this step
            selected_logits = self._gather_beams(logits_new_token, top_beam_indices)
            logits_token_in_beam = torch.cat([
                self._gather_beams(logits_token_in_beam, top_beam_indices),
                selected_logits.unsqueeze(2)
            ], dim=2)
        
        return beams.long(), logits_token_in_beam
    
    def _gather_beams(self, x: torch.Tensor, selected_beams: torch.Tensor) -> torch.Tensor:
        """
        Gather selected beams from tensor x.
        
        Args:
            x: (batch_size, beam_width, ...) Input tensor
            selected_beams: (batch_size, beam_width) Beam indices to select
            
        Returns:
            Selected beams with same shape as input
        """
        batch_size, beam_width = x.shape[0], x.shape[1]
        batch_indices = torch.arange(batch_size * beam_width, device=x.device) // beam_width
        batch_indices = batch_indices.reshape(batch_size, beam_width)
        return x[batch_indices, selected_beams]
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[Any] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        beam_width: int = 10,
        beam_length: int = 4,
        temperature: float = 1.0,
        greedy: bool = False,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Generate text using recurrent drafting with beam search.
        
        This method implements the complete speculative decoding loop with
        sophisticated candidate generation, verification, and acceptance.
        
        Args:
            inputs: Input token IDs [batch_size, seq_len]
            generation_config: Generation configuration (for compatibility)
            max_length: Maximum total sequence length
            max_new_tokens: Maximum number of new tokens to generate
            beam_width: Number of beams for candidate search
            beam_length: Length of each candidate beam
            temperature: Sampling temperature for probabilistic acceptance
            greedy: Whether to use greedy verification (exact matching)
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            **kwargs: Additional generation arguments
            
        Returns:
            Generated token sequences [batch_size, new_seq_len]
        """
        if self.verifier is None:
            raise ValueError(
                "Verifier model must be attached for generation. "
                "Use model.attach_verifier(verifier_model) first."
            )
        
        # Handle input arguments
        if inputs is None:
            inputs = kwargs.get('input_ids')
        if inputs is None:
            raise ValueError("Must provide input_ids for generation")
        
        # Determine generation limits
        batch_size, initial_seq_len = inputs.shape
        device = inputs.device
        
        if max_length is None and max_new_tokens is None:
            max_length = initial_seq_len + 50  # Default to 50 new tokens
        elif max_length is None:
            max_length = initial_seq_len + max_new_tokens
        
        # Get token IDs from verifier config if not provided
        if pad_token_id is None:
            pad_token_id = getattr(self.verifier.config, 'pad_token_id', 0)
        if eos_token_id is None:
            eos_token_id = getattr(self.verifier.config, 'eos_token_id', None)
        
        # Initialize generation state
        input_ids = inputs.clone()
        beam_shape = BeamShape(beam_width, beam_length)
        generation_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Get initial context from verifier
        with torch.no_grad():
            verifier_outputs = self.verifier(input_ids, output_hidden_states=True)
            prompt_state = verifier_outputs.hidden_states[-1][:, -1, :]  # Last token, last layer
            
            # Sample initial token from verifier
            next_token_logits = verifier_outputs.logits[:, -1, :]
            if greedy:
                init_token = torch.argmax(next_token_logits, dim=-1)
            else:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                init_token = torch.multinomial(probs, 1).squeeze(-1)
        
        # Main generation loop
        step_count = 0
        max_steps = max_length - initial_seq_len + 10  # Safety margin
        
        while input_ids.shape[1] < max_length and not generation_finished.all() and step_count < max_steps:
            step_count += 1
            
            # Skip finished sequences
            if generation_finished.all():
                break
            
            try:
                # Generate candidate beams using the drafter
                beams, log_probs_by_drafter = self.beam_search_candidates(
                    prompt_state, init_token, self.verifier.get_input_embeddings(), beam_shape
                )
                
                # Verify candidates with the base model
                hidden_states, logits_by_llm = self._verify_candidates(input_ids, beams)
                
                # Accept/reject candidates based on verification method
                if greedy:
                    prompt_state, input_ids, init_token, n_accepted = self._greedy_accept_candidates(
                        input_ids, beams, logits_by_llm, hidden_states
                    )
                else:
                    prompt_state, input_ids, init_token, n_accepted = self._accept_candidates(
                        input_ids, beams, log_probs_by_drafter, logits_by_llm, 
                        hidden_states, temperature
                    )
                
                # Check for EOS tokens if specified
                if eos_token_id is not None:
                    # Check if any sequence has generated EOS
                    for i in range(batch_size):
                        if not generation_finished[i]:
                            # Check if EOS token was just generated
                            if init_token[i] == eos_token_id:
                                generation_finished[i] = True
                            # Also check in the accepted tokens
                            elif n_accepted[i] > 0:
                                recent_tokens = input_ids[i, -n_accepted[i].item():]
                                if eos_token_id in recent_tokens:
                                    generation_finished[i] = True
                
                # Check if we've made progress (avoid infinite loops)
                if n_accepted.sum() == 0:
                    # No tokens accepted, generate one token directly from verifier
                    with torch.no_grad():
                        verifier_outputs = self.verifier(input_ids, output_hidden_states=True)
                        next_token_logits = verifier_outputs.logits[:, -1, :]
                        
                        if greedy:
                            next_token = torch.argmax(next_token_logits, dim=-1)
                        else:
                            probs = torch.softmax(next_token_logits / temperature, dim=-1)
                            next_token = torch.multinomial(probs, 1).squeeze(-1)
                        
                        # Update sequences
                        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=1)
                        prompt_state = verifier_outputs.hidden_states[-1][:, -1, :]
                        init_token = next_token
                
            except Exception as e:
                # Handle any errors gracefully
                print(f"Warning: Error in generation step {step_count}: {e}")
                # Fall back to single token generation
                with torch.no_grad():
                    verifier_outputs = self.verifier(input_ids, output_hidden_states=True)
                    next_token_logits = verifier_outputs.logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1)
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=1)
                    prompt_state = verifier_outputs.hidden_states[-1][:, -1, :]
                    init_token = next_token
        
        # Truncate to max_length if necessary
        if input_ids.shape[1] > max_length:
            input_ids = input_ids[:, :max_length]
        
        return input_ids
    
    def _verify_candidates(
        self, input_ids: torch.Tensor, beams: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Verify candidate beams using the base model.
        
        This method processes all candidate sequences through the verifier model
        to get their logits and hidden states for acceptance/rejection decisions.
        
        Args:
            input_ids: Current input sequence [batch_size, seq_len]
            beams: Candidate beam sequences [batch_size, beam_width, beam_length]
            
        Returns:
            hidden_states: Hidden states from verification [batch_size, beam_width, beam_length, hidden_size]
            logits: Logits from verification [batch_size, beam_width, beam_length, vocab_size]
        """
        batch_size, beam_width, beam_length = beams.shape
        
        # Flatten beams for efficient batch processing
        flat_beams = beams.view(-1, beam_length)
        
        # Create input sequences by concatenating input_ids with each beam
        expanded_inputs = input_ids.unsqueeze(1).repeat(1, beam_width, 1)
        expanded_inputs = expanded_inputs.view(-1, input_ids.shape[1])
        
        # Concatenate current sequence with candidate continuations
        full_sequences = torch.cat([expanded_inputs, flat_beams], dim=1)
        
        # Run through verifier model to get predictions
        with torch.no_grad():
            outputs = self.verifier(full_sequences, output_hidden_states=True)
        
        # Extract only the logits for the candidate tokens (not the input sequence)
        candidate_logits = outputs.logits[:, -beam_length:, :]
        candidate_hidden_states = outputs.hidden_states[-1][:, -beam_length:, :]
        
        # Reshape outputs back to beam structure
        logits = candidate_logits.view(batch_size, beam_width, beam_length, -1)
        hidden_states = candidate_hidden_states.view(batch_size, beam_width, beam_length, -1)
        
        return hidden_states, logits
    
    def _greedy_accept_candidates(
        self, input_ids: torch.Tensor, beams: torch.Tensor, 
        logits_by_llm: torch.Tensor, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Greedy candidate acceptance using exact token matching.
        
        This method compares the drafter's candidate tokens with the verifier's
        top-1 predictions and accepts the longest matching prefix from the best beam.
        
        Args:
            input_ids: Current input sequence [batch_size, seq_len]
            beams: Candidate sequences from drafter [batch_size, beam_width, beam_length]
            logits_by_llm: Verifier logits [batch_size, beam_width, beam_length, vocab_size]
            hidden_states: Verifier hidden states [batch_size, beam_width, beam_length, hidden_size]
            
        Returns:
            last_token_state: Hidden state of last accepted token [batch_size, hidden_size]
            updated_input_ids: Input sequence with accepted tokens [batch_size, new_seq_len]
            next_tokens: Next token for continued generation [batch_size]
            n_tokens_accepted: Number of tokens accepted per batch item [batch_size]
        """
        batch_size, beam_width, beam_length = beams.shape
        
        # Get verifier's top-1 predictions for each position
        beams_by_llm = torch.argmax(logits_by_llm, dim=-1)
        
        # Compare drafter predictions with verifier predictions
        # Note: beams include the init token, so we compare beams[1:] with verifier predictions
        drafter_tokens = beams[:, :, 1:]  # Skip init token
        verifier_tokens = beams_by_llm[:, :, :-1]  # Skip last prediction (for next token)
        
        # Find matching tokens
        matches = (drafter_tokens == verifier_tokens)
        
        # Find longest matching prefix for each beam using cumulative product
        # This gives us the number of consecutive matches from the start
        cumulative_matches = torch.cumprod(matches, dim=-1)
        n_tokens_in_seqs = torch.sum(cumulative_matches, dim=-1)
        
        # Select the beam with the most accepted tokens for each batch item
        n_tokens_accepted, best_beam_indices = torch.max(n_tokens_in_seqs, dim=-1)
        
        # Extract accepted tokens from the best beams
        batch_indices = torch.arange(batch_size, device=beams.device)
        best_beams = beams[batch_indices, best_beam_indices]  # [batch_size, beam_length]
        
        # Get the accepted portion of each beam (excluding init token, including only accepted tokens)
        accepted_tokens_list = []
        for i in range(batch_size):
            n_accepted = n_tokens_accepted[i].item()
            if n_accepted > 0:
                # Take tokens 1 to n_accepted+1 (skip init token at position 0)
                accepted = best_beams[i, 1:n_accepted+1]
                accepted_tokens_list.append(accepted)
            else:
                # No tokens accepted, create empty tensor
                accepted_tokens_list.append(torch.empty(0, dtype=beams.dtype, device=beams.device))
        
        # Pad accepted tokens to same length for batching
        if any(len(tokens) > 0 for tokens in accepted_tokens_list):
            max_accepted = max(len(tokens) for tokens in accepted_tokens_list)
            padded_accepted = torch.zeros(batch_size, max_accepted, dtype=beams.dtype, device=beams.device)
            for i, tokens in enumerate(accepted_tokens_list):
                if len(tokens) > 0:
                    padded_accepted[i, :len(tokens)] = tokens
            
            # Update input_ids with accepted tokens
            updated_input_ids = torch.cat([input_ids, padded_accepted], dim=1)
        else:
            # No tokens accepted in any batch item
            updated_input_ids = input_ids
        
        # Get next token from verifier's predictions at the rejection point
        next_tokens = torch.zeros(batch_size, dtype=torch.long, device=beams.device)
        last_token_states = torch.zeros(batch_size, hidden_states.shape[-1], device=beams.device)
        
        for i in range(batch_size):
            beam_idx = best_beam_indices[i]
            n_accepted = n_tokens_accepted[i].item()
            
            # Next token is the verifier's prediction at the rejection point
            next_tokens[i] = beams_by_llm[i, beam_idx, n_accepted]
            
            # Last token state is from the last accepted position
            if n_accepted > 0:
                last_token_states[i] = hidden_states[i, beam_idx, n_accepted - 1]
            else:
                # No tokens accepted, use the state from the init token position
                last_token_states[i] = hidden_states[i, beam_idx, 0]
        
        return last_token_states, updated_input_ids, next_tokens, n_tokens_accepted
    
    def _accept_candidates(
        self, input_ids: torch.Tensor, beams: torch.Tensor,
        log_probs_by_drafter: torch.Tensor, logits_by_llm: torch.Tensor,
        hidden_states: torch.Tensor, temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Probabilistic candidate acceptance using rejection sampling.
        
        This method implements the sophisticated rejection sampling approach from
        the original recurrent drafting paper, comparing drafter and verifier
        probabilities to make acceptance decisions.
        
        Args:
            input_ids: Current input sequence
            beams: Candidate sequences from drafter
            log_probs_by_drafter: Log probabilities from drafter
            logits_by_llm: Logits from verifier
            hidden_states: Hidden states from verifier
            temperature: Sampling temperature
            
        Returns:
            Same as _greedy_accept_candidates
        """
        batch_size, beam_width, beam_length = beams.shape
        
        # Convert verifier logits to log probabilities
        log_probs_by_llm = torch.log_softmax(logits_by_llm / temperature, dim=-1)
        
        # Extract the tokens that were actually generated (excluding init token)
        drafted_tokens = beams[:, :, 1:]  # [batch_size, beam_width, beam_length-1]
        
        # Get log probabilities for the drafted tokens from both models
        # We need to gather the probabilities for the specific tokens that were drafted
        drafted_tokens_expanded = drafted_tokens.unsqueeze(-1)  # Add vocab dimension for gathering
        
        # Get drafter probabilities for drafted tokens
        q_probs = torch.gather(log_probs_by_drafter, dim=-1, index=drafted_tokens_expanded).squeeze(-1)
        
        # Get verifier probabilities for drafted tokens (excluding last position which is for next token)
        p_probs = torch.gather(log_probs_by_llm[:, :, :-1, :], dim=-1, index=drafted_tokens_expanded).squeeze(-1)
        
        # Rejection sampling: accept if random < min(1, p/q) = min(1, exp(log_p - log_q))
        # This is equivalent to: accept if random < exp(log_p - log_q) when p < q, always accept when p >= q
        log_acceptance_prob = torch.clamp(p_probs - q_probs, max=0.0)  # min(0, log_p - log_q) = log(min(1, p/q))
        acceptance_prob = torch.exp(log_acceptance_prob)
        
        # Generate random numbers for rejection sampling
        random_vals = torch.rand_like(acceptance_prob)
        accept_mask = random_vals < acceptance_prob
        
        # Find the longest accepted prefix for each beam
        # Use cumulative product to find consecutive acceptances from the start
        cumulative_accepts = torch.cumprod(accept_mask, dim=-1)
        n_tokens_in_seqs = torch.sum(cumulative_accepts, dim=-1)
        
        # Select the beam with the most accepted tokens
        n_tokens_accepted, best_beam_indices = torch.max(n_tokens_in_seqs, dim=-1)
        
        # Extract accepted tokens and update sequences (similar to greedy method)
        batch_indices = torch.arange(batch_size, device=beams.device)
        best_beams = beams[batch_indices, best_beam_indices]
        
        # Build accepted tokens list
        accepted_tokens_list = []
        for i in range(batch_size):
            n_accepted = n_tokens_accepted[i].item()
            if n_accepted > 0:
                accepted = best_beams[i, 1:n_accepted+1]  # Skip init token
                accepted_tokens_list.append(accepted)
            else:
                accepted_tokens_list.append(torch.empty(0, dtype=beams.dtype, device=beams.device))
        
        # Update input_ids with accepted tokens
        if any(len(tokens) > 0 for tokens in accepted_tokens_list):
            max_accepted = max(len(tokens) for tokens in accepted_tokens_list)
            padded_accepted = torch.zeros(batch_size, max_accepted, dtype=beams.dtype, device=beams.device)
            for i, tokens in enumerate(accepted_tokens_list):
                if len(tokens) > 0:
                    padded_accepted[i, :len(tokens)] = tokens
            updated_input_ids = torch.cat([input_ids, padded_accepted], dim=1)
        else:
            updated_input_ids = input_ids
        
        # For next token generation, we need to sample from the corrected distribution
        # at the rejection point using rejection sampling
        next_tokens = torch.zeros(batch_size, dtype=torch.long, device=beams.device)
        last_token_states = torch.zeros(batch_size, hidden_states.shape[-1], device=beams.device)
        
        for i in range(batch_size):
            beam_idx = best_beam_indices[i]
            n_accepted = n_tokens_accepted[i].item()
            
            # Get the corrected probability distribution at the rejection point
            if n_accepted < beam_length - 1:
                # There was a rejection, use corrected sampling
                p_dist = torch.softmax(log_probs_by_llm[i, beam_idx, n_accepted] / temperature, dim=-1)
                q_dist = torch.softmax(log_probs_by_drafter[i, beam_idx, n_accepted], dim=-1)
                
                # Corrected distribution: max(0, p - q) normalized
                corrected_dist = torch.clamp(p_dist - q_dist, min=0.0)
                corrected_dist = corrected_dist / (corrected_dist.sum() + 1e-10)  # Normalize
                
                # Sample from corrected distribution
                next_tokens[i] = torch.multinomial(corrected_dist, 1).squeeze()
            else:
                # All tokens were accepted, sample from verifier's next token distribution
                p_dist = torch.softmax(log_probs_by_llm[i, beam_idx, -1] / temperature, dim=-1)
                next_tokens[i] = torch.multinomial(p_dist, 1).squeeze()
            
            # Get last token state
            if n_accepted > 0:
                last_token_states[i] = hidden_states[i, beam_idx, n_accepted - 1]
            else:
                last_token_states[i] = hidden_states[i, beam_idx, 0]
        
        return last_token_states, updated_input_ids, next_tokens, n_tokens_accepted
    
