
import os
import torch
import torch.nn.functional as F
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Dict, Optional


class SpeculativeDecoder:
    def __init__(self, target_model_name: str, draft_model_name: str, device: str = "cuda:0"):
        """
        Initialize the speculative decoder with target and draft models.

        Args:
            target_model_name: HuggingFace model ID for the larger target model.
            draft_model_name: HuggingFace model ID for the smaller draft model.
            device: Device to run models on ("cuda" or "cpu").
        """
        self.device = device
        self.target_model, self.target_tokenizer = self.initialize_target_model(target_model_name)
        self.draft_model, self.draft_tokenizer = self.initialize_draft_model(draft_model_name)

        # Ensure tokenizers are compatible
        assert self.target_tokenizer.vocab == self.draft_tokenizer.vocab, "Tokenizers must be compatible"

    def initialize_target_model(self, model_name: str):
        """Initialize the larger target model with caching enabled and proper pad token."""
        print(f"Loading target model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print("Pad token set to EOS token")

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cuda:0")
        model.eval()
        model.to(self.device)

        # TODO: Implement target model initialization
        # 1. Set the pad token if it doesn't exist
        # 2. Load the model with appropriate settings for inference
        # 3. Enable any optimizations that might help with performance

        return model, tokenizer

    def initialize_draft_model(self, model_name: str):
        """
        Initialize a smaller, faster draft model with proper pad token.
        Uses lower precision and additional optimizations.
        """
        print(f"Loading draft model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print("Pad token set to EOS token")

        # TODO: Implement draft model initialization
        # 1. Set the pad token if it doesn't exist
        # 2. Load the model with appropriate settings for inference
        # 3. Enable any optimizations that might help with performance

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cuda:0")
        model.eval()
        model.to(self.device)

        return model, tokenizer

    def generate_draft_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                             num_speculative_tokens: int = 10) -> torch.Tensor:
        """
        Generate speculative tokens in one forward call using the draft model.

        Args:
            input_ids: Input token IDs (tensor of shape [1, seq_len]).
            attention_mask: Corresponding attention mask.
            num_speculative_tokens: Number of tokens to speculate.

        Returns:
            Tensor of shape [1, num_speculative_tokens] containing the draft tokens.
        """
        # TODO: Implement draft token generation
        # 1. Use the draft model to generate tokens
        # 2. Extract only the new tokens (not including the input)
        # 3. Return the newly generated tokens
        # print(f"input_ids: {input_ids}")
        with torch.no_grad():
            draft_ids = self.draft_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + num_speculative_tokens,
                do_sample=False,
                pad_token_id=self.draft_tokenizer.pad_token_id,
                eos_token_id=self.draft_tokenizer.eos_token_id,
            )

        new_tokens = draft_ids[:, input_ids.shape[1]:]
        return new_tokens
    

    def verify_tokens_vectorized(self, input_ids: torch.Tensor, draft_ids: torch.Tensor,
                               attention_mask: torch.Tensor) -> Tuple[List[int], int]:
        """
        Vectorized verification: verify all draft tokens in one forward pass using the target model.

        Args:
            input_ids: The current input token IDs (shape [1, L]).
            draft_tokens: Draft tokens from the draft model (shape [1, k]).
            attention_mask: The current attention mask for input_ids.

        Returns:
            accepted_tokens: List of accepted token IDs.
            accepted_position: Index of the first rejected token (if all accepted, equals draft_tokens.shape[1]).
        """
        # TODO: Implement efficient verification of draft tokens
        # 1. Run target model on input_ids concatenated with draft_tokens
        # 2. Extract the logits for positions where draft tokens would be predicted
        # 3. Compare target model predictions with draft tokens
        # 4. Determine how many consecutive tokens were accepted before first mismatch

        # print(f"input_ids: {input_ids}")
        # print(f"current attention mask: {attention_mask}")
       

        # print(f"decoded draft tokens: {self.draft_tokenizer.decode(draft_model_outputs.sequences[0], skip_special_tokens=True)}")
        concat_input = torch.cat([input_ids, draft_ids], dim=1)
        
        with torch.no_grad():
            target_model_output = self.target_model(
                input_ids=concat_input,
                attention_mask=torch.cat([attention_mask, attention_mask.new_ones(draft_ids.shape)], dim=1),
            )

        # print(f"target model output: {target_model_output}")
        # print(f"target logits shape: {target_model_output.logits.shape}")

        target_logits = target_model_output.logits[:, input_ids.shape[1]-1: -1, :]
        target_id = torch.argmax(target_logits, dim=-1)

        accepted_position = 0
        for i in range(target_id.shape[1]):
            if target_id[:, i] == draft_ids[:, i]:
                accepted_position += 1
            else:
                break
           
        return target_id[:, :accepted_position], accepted_position
    
    def speculative_decode(self, prompt: str, max_tokens: int = 100,
                          num_speculative_tokens: int = 5) -> str:
        """
        Main speculative decoding algorithm with vectorized verification.

        Args:
            prompt: Input text.
            max_tokens: Maximum number of tokens to generate (excluding prompt).
            num_speculative_tokens: Number of tokens to speculate per iteration.

        Returns:
            Generated text.
        """
        # Tokenize prompt
        inputs = self.target_tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        prompt_length = input_ids.shape[1]

        # Initialize counters for performance tracking
        total_tokens_generated = prompt_length
        total_draft_tokens_proposed = 0
        total_draft_tokens_accepted = 0
        start_time = time.time()

        # TODO: Implement the core speculative decoding loop
        # 1. Generate draft tokens using the draft model
        # 2. Verify draft tokens using the target model
        # 3. Accept verified tokens and append to the sequence
        # 4. For rejected tokens or if all tokens are accepted, generate a new token with the target model
        # 5. Stop when max_tokens is reached or an EOS token is generated

        while total_tokens_generated < max_tokens + prompt_length:
            draft_ids = self.generate_draft_tokens(input_ids, attention_mask, num_speculative_tokens)
           
            total_draft_tokens_proposed += draft_ids.shape[1]

            accepted_tokens, accepted_position = self.verify_tokens_vectorized(input_ids, draft_ids, attention_mask)

            total_draft_tokens_accepted += accepted_position
            total_tokens_generated += accepted_position
            
            if accepted_position == 0:
                target_input_ids = input_ids
            else:
                target_input_ids = torch.cat([input_ids, accepted_tokens], dim=1)

            if accepted_position == draft_ids.shape[1]:
                # All draft tokens accepted, generate next token with target model
                input_ids = target_input_ids
                attention_mask = torch.ones_like(input_ids)
            else:
                with torch.no_grad():
                    target_model_output = self.target_model.generate(
                        input_ids=target_input_ids,
                        attention_mask=torch.ones_like(target_input_ids),
                        max_new_tokens=1,
                        do_sample=False,
                        pad_token_id=self.target_tokenizer.pad_token_id,
                    )

                total_tokens_generated += 1
                input_ids = target_model_output
                attention_mask = torch.ones_like(input_ids)  
    

        # Calculate performance metrics
        elapsed_time = time.time() - start_time
        acceptance_rate = total_draft_tokens_accepted / total_draft_tokens_proposed if total_draft_tokens_proposed > 0 else 0

        print(f"Generated {total_tokens_generated - prompt_length} tokens in {elapsed_time:.2f} seconds")
        print(f"Tokens per second: {(total_tokens_generated - prompt_length) / elapsed_time:.2f}")
        print(f"Draft token acceptance rate: {acceptance_rate:.2%}")

        return self.target_tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def benchmark(self, prompt: str, max_tokens: int = 100,
                  num_runs: int = 3, compare_baseline: bool = True) -> Dict:
        """
        Benchmark the speculative decoder against baseline decoding.

        Args:
            prompt: Input text.
            max_tokens: Maximum number of tokens to generate.
            num_runs: Number of benchmark runs.
            compare_baseline: Whether to compare with baseline (non-speculative) decoding.

        Returns:
            Dictionary with benchmark results.
        """
        results = {
            "speculative": {"times": [], "tokens_per_second": []},
            "baseline": {"times": [], "tokens_per_second": []} if compare_baseline else None
        }

        # Benchmark speculative decoding.
        for _ in range(num_runs):
            start_time = time.time()
            output = self.speculative_decode(prompt, max_tokens=max_tokens)
            elapsed = time.time() - start_time
            prompt_len = len(self.target_tokenizer(prompt)["input_ids"])
            output_tokens = len(self.target_tokenizer.encode(output)) - prompt_len
            tps = output_tokens / elapsed
            results["speculative"]["times"].append(elapsed)
            results["speculative"]["tokens_per_second"].append(tps)

        # Benchmark baseline decoding.
        if compare_baseline:
            for _ in range(num_runs):
                inputs = self.target_tokenizer(prompt, return_tensors="pt", padding=True)
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                start_time = time.time()
                with torch.no_grad():
                    output_ids = self.target_model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.shape[1] + max_tokens,
                        do_sample=False,
                        pad_token_id=self.target_tokenizer.pad_token_id,
                        return_dict_in_generate=True
                    )
                elapsed = time.time() - start_time
                output_tokens = output_ids.sequences.shape[1] - input_ids.shape[1]
                tps = output_tokens / elapsed
                results["baseline"]["times"].append(elapsed)
                results["baseline"]["tokens_per_second"].append(tps)

        for method in results.keys():
            if results[method] is not None:
                avg_time = sum(results[method]["times"]) / num_runs
                avg_tps = sum(results[method]["tokens_per_second"]) / num_runs
                results[method]["avg_time"] = avg_time
                results[method]["avg_tokens_per_second"] = avg_tps

        if compare_baseline:
            speedup = results["baseline"]["avg_time"] / results["speculative"]["avg_time"]
            results["speedup"] = speedup
            results["latency_reduction"] = (1 - results["speculative"]["avg_time"] / results["baseline"]["avg_time"]) * 100
            # print(f"Speculative decoding speedup: {speedup:.2f}x")
            # print(f"Latency reduction: {results['latency_reduction']:.2f}%")

        return results

if __name__ == "__main__":
    print("Running spec_decode.py directly")