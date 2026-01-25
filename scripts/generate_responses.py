"""
Generate model responses for all rollouts using vLLM for efficient batched inference.

Features:
- Batched inference using vLLM for speed
- Resume capability (tracks processed indices)
- Incremental saving to avoid data loss
- Progress tracking with tqdm

Usage:
    python scripts/generate_responses.py
    python scripts/generate_responses.py --batch-size 64 --save-every 1000
    python scripts/generate_responses.py --model google/gemma-2-9b-it  # Use larger model
    python scripts/generate_responses.py --no-resume  # Start fresh
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import time

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("ERROR: vLLM is not installed.")
    print("\nTo install vLLM:")
    print("  pip install vllm")
    print("\nNote: vLLM requires:")
    print("  - CUDA-compatible GPU")
    print("  - Linux (or WSL on Windows)")
    print("  - Python 3.8+")
    exit(1)


def load_rollouts(rollouts_path: Path) -> pd.DataFrame:
    """Load rollouts from parquet file."""
    print(f"Loading rollouts from {rollouts_path}...")
    df = pd.read_parquet(rollouts_path)
    print(f"Loaded {len(df):,} rollouts")
    return df


def load_existing_responses(responses_path: Path) -> pd.DataFrame:
    """Load existing responses if they exist."""
    if responses_path.exists():
        print(f"Loading existing responses from {responses_path}...")
        df = pd.read_parquet(responses_path)
        print(f"Found {len(df):,} existing responses")
        return df
    else:
        print("No existing responses found, starting fresh")
        return pd.DataFrame()


def get_unprocessed_indices(rollouts_df: pd.DataFrame, responses_df: pd.DataFrame) -> List[int]:
    """Determine which rollout indices haven't been processed yet."""
    if responses_df.empty:
        return list(range(len(rollouts_df)))

    # Assume responses_df has a 'rollout_idx' column
    processed_indices = set(responses_df['rollout_idx'].values)
    all_indices = set(range(len(rollouts_df)))
    unprocessed = sorted(all_indices - processed_indices)

    print(f"Already processed: {len(processed_indices):,}")
    print(f"Remaining: {len(unprocessed):,}")

    return unprocessed


def construct_prompt(instruction: str, question: str) -> str:
    """Construct the prompt from instruction and question.

    The instruction is the system prompt that tells the model to play a role.
    The question is what we ask the model.
    """
    # For Gemma, we use the chat template format
    # Instruction goes in system message, question in user message
    prompt = f"{instruction}\n\n{question}"
    return prompt


def generate_responses_batch(
    llm: LLM,
    rollouts: List[Dict],
    indices: List[int],
    sampling_params: SamplingParams,
    model_name: str
) -> List[Dict]:
    """Generate responses for a batch of rollouts.

    Args:
        llm: vLLM model
        rollouts: List of rollout dicts with 'instruction_text' and 'question_text'
        indices: List of indices in the original rollouts dataframe
        sampling_params: vLLM sampling parameters
        model_name: Name of the model being used

    Returns:
        List of dicts with rollout_idx, prompt, and response
    """
    # Construct prompts
    prompts = [
        construct_prompt(r['instruction_text'], r['question_text'])
        for r in rollouts
    ]

    # Generate responses using vLLM
    outputs = llm.generate(prompts, sampling_params)

    # Extract results
    results = []
    for idx, rollout, output in zip(indices, rollouts, outputs):
        results.append({
            'rollout_idx': idx,
            'model': model_name,
            'role': rollout['role'],
            'role_category': rollout['role_category'],
            'instruction_idx': rollout['instruction_idx'],
            'question_idx': rollout['question_idx'],
            'prompt': output.prompt,
            'response': output.outputs[0].text,
            'finish_reason': output.outputs[0].finish_reason,
            'tokens_generated': len(output.outputs[0].token_ids)
        })

    return results


def save_responses(responses: List[Dict], output_path: Path, mode='append'):
    """Save responses to parquet file.

    Args:
        responses: List of response dicts
        output_path: Path to save responses
        mode: 'append' to add to existing file, 'overwrite' to replace
    """
    new_df = pd.DataFrame(responses)

    if mode == 'append' and output_path.exists():
        # Load existing and append
        existing_df = pd.read_parquet(output_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_parquet(output_path, compression='snappy', index=False)
    else:
        # Write new file
        new_df.to_parquet(output_path, compression='snappy', index=False)

    print(f"Saved {len(responses)} responses to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate model responses for rollouts')
    parser.add_argument('--model', type=str, default='google/gemma-2-2b-it',
                        help='Model name/path to use (default: google/gemma-2-2b-it). '
                             'For larger model: google/gemma-2-9b-it (requires ~17GB VRAM)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for inference (default: 64)')
    parser.add_argument('--save-every', type=int, default=500,
                        help='Save responses every N rollouts (default: 500)')
    parser.add_argument('--max-tokens', type=int, default=512,
                        help='Maximum tokens to generate (default: 512)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (default: 0.7)')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                        help='Fraction of GPU memory to use (default: 0.9)')
    parser.add_argument('--max-model-len', type=int, default=2048,
                        help='Maximum model context length (default: 2048)')
    parser.add_argument('--quantization', type=str, default=None,
                        help='Quantization method (awq, gptq, or None for no quantization)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh, ignoring existing responses')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of rollouts to process (for testing)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file name (default: responses_<model_name>.parquet)')

    args = parser.parse_args()

    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    rollouts_path = data_dir / "rollouts.parquet"

    # Generate output filename from model name if not specified
    if args.output:
        responses_path = data_dir / args.output
    else:
        # Extract model name from path (e.g., "google/gemma-2-2b-it" -> "gemma-2-2b-it")
        model_name = args.model.split("/")[-1]
        responses_path = data_dir / f"responses_{model_name}.parquet"

    # Load rollouts
    rollouts_df = load_rollouts(rollouts_path)
    print(f"Output will be saved to: {responses_path}")

    # Determine what needs to be processed
    if args.no_resume:
        print("Starting fresh (--no-resume flag)")
        unprocessed_indices = list(range(len(rollouts_df)))
        if responses_path.exists():
            responses_path.unlink()
            print(f"Deleted existing {responses_path}")
    else:
        responses_df = load_existing_responses(responses_path)
        unprocessed_indices = get_unprocessed_indices(rollouts_df, responses_df)

    if not unprocessed_indices:
        print("All rollouts have been processed!")
        return

    # Apply limit if specified
    if args.limit:
        unprocessed_indices = unprocessed_indices[:args.limit]
        print(f"Limited to {args.limit} rollouts for testing")

    # Initialize vLLM
    print(f"\nInitializing vLLM with model: {args.model}")
    print("This may take a few minutes to download and load the model...")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"Max model length: {args.max_model_len}")
    if args.quantization:
        print(f"Quantization: {args.quantization}")

    # Build vLLM arguments
    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": 1,  # Use 1 GPU
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
    }

    # Add quantization if specified
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization

    llm = LLM(**llm_kwargs)

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=0.9,
    )

    print(f"\nGeneration parameters:")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Save every: {args.save_every} rollouts")

    # Process in batches
    all_responses = []
    batch_indices = []
    batch_rollouts = []

    print(f"\nProcessing {len(unprocessed_indices):,} rollouts...")

    start_time = time.time()
    last_save_time = start_time

    with tqdm(total=len(unprocessed_indices), desc="Generating responses") as pbar:
        for i, idx in enumerate(unprocessed_indices):
            # Get rollout data
            rollout = rollouts_df.iloc[idx].to_dict()

            batch_indices.append(idx)
            batch_rollouts.append(rollout)

            # Process batch when full or at end
            if len(batch_rollouts) >= args.batch_size or i == len(unprocessed_indices) - 1:
                # Generate responses for this batch
                batch_results = generate_responses_batch(
                    llm, batch_rollouts, batch_indices, sampling_params, args.model
                )

                all_responses.extend(batch_results)

                # Update progress
                pbar.update(len(batch_rollouts))

                # Save periodically
                if len(all_responses) >= args.save_every or i == len(unprocessed_indices) - 1:
                    save_responses(all_responses, responses_path, mode='append')

                    # Calculate stats
                    elapsed = time.time() - start_time
                    processed = i + 1
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = len(unprocessed_indices) - processed
                    eta = remaining / rate if rate > 0 else 0

                    print(f"\nProgress: {processed:,}/{len(unprocessed_indices):,} "
                          f"({100*processed/len(unprocessed_indices):.1f}%)")
                    print(f"Rate: {rate:.1f} rollouts/sec")
                    print(f"ETA: {eta/60:.1f} minutes")

                    all_responses = []  # Clear buffer after saving
                    last_save_time = time.time()

                # Reset batch
                batch_indices = []
                batch_rollouts = []

    # Final statistics
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✓ Response generation complete!")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total processed: {len(unprocessed_indices):,} rollouts")
    print(f"Average rate: {len(unprocessed_indices)/total_time:.1f} rollouts/sec")
    print(f"\nResponses saved to: {responses_path}")

    # Load and show final stats
    final_df = pd.read_parquet(responses_path)
    print(f"\nFinal response count: {len(final_df):,}")
    print(f"Average response length: {final_df['tokens_generated'].mean():.1f} tokens")


if __name__ == "__main__":
    main()
