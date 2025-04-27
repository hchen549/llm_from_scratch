import torch

from spec_decoding.spec_decode import SpeculativeDecoder

target_model_name = "EleutherAI/pythia-1.4b-deduped"  # Larger target model
draft_model_name = "EleutherAI/pythia-160m-deduped"   # Smaller draft model


# Initialize speculative decoder
decoder = SpeculativeDecoder(
    target_model_name=target_model_name,
    draft_model_name=draft_model_name,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Test prompts
test_prompts = [
    "The future of Artificial Intelligence is",
    "Write a short story about a robot learning to feel emotions:",
    "Write the lyrics to the song 'Happy Birthday'."
]

# Run benchmark on test prompts
for i, prompt in enumerate(test_prompts):
    print(f"\nBenchmarking Prompt {i+1}:")
    print(f"Prompt: {prompt}")

    results = decoder.benchmark(
        prompt=prompt,
        max_tokens=100,
        num_runs=3,
        compare_baseline=True
    )

    print(f"Average speculative decoding time: {results['speculative']['avg_time']:.2f} seconds")
    print(f"Average speculative tokens per second: {results['speculative']['avg_tokens_per_second']:.2f}")

    if results["baseline"] is not None:
        print(f"Average baseline decoding time: {results['baseline']['avg_time']:.2f} seconds")
        print(f"Average baseline tokens per second: {results['baseline']['avg_tokens_per_second']:.2f}")
        print(f"Speedup: {results['speedup']:.2f}x")
        print(f"Latency reduction: {results['latency_reduction']:.2f}%")