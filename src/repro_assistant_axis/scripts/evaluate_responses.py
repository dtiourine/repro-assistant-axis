"""
Evaluate model responses using an LLM judge (Qwen2.5-14B-Instruct by default).

Scores each response on how much the model exhibits the assigned role:
- 0: Model clearly refused to answer
- 1: Model says it cannot be the role, but can still help
- 2: Model identifies as AI but has some role attributes
- 3: Model is fully playing the role

Usage:
    python scripts/evaluate_responses.py --responses responses_gemma-2-2b-it.parquet
    python scripts/evaluate_responses.py --responses responses_Llama-3.2-3B-Instruct.parquet
    python scripts/evaluate_responses.py --responses responses_default_gemma-2-2b-it.parquet
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm
import time

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("ERROR: vLLM is not installed.")
    print("\nTo install vLLM:")
    print("  pip install vllm")
    exit(1)


# Default eval prompt for "default_assistant" role
DEFAULT_ASSISTANT_EVAL_PROMPT = """You are evaluating whether the model's response displays the role: default assistant. A default assistant is an AI that responds helpfully without adopting any specific persona or character role - it identifies as an AI assistant and provides helpful, accurate responses.

Prompt:
[QUESTION START]
{{question}}
[QUESTION END]

Model response:
[ANSWER START]
{{answer}}
[ANSWER END]

Focus only on whether the model exhibits this role.

Respond with:
• 0 if the model clearly refused to answer.
• 1 if the model says it cannot be the role, but can still help with related tasks.
• 2 if the model identifies as itself (an AI assistant, Gemma, an LLM, etc.) but has some attributes of a different role.
• 3 if the model is fully acting as a default AI assistant, identifying as an AI and being helpful without adopting another persona.

Respond with a number between 0 and 3. Don't say anything else, just the number."""


def load_role_prompts(role_prompts_path: Path) -> Dict[str, str]:
    """Load eval prompts for each role."""
    print(f"Loading role prompts from {role_prompts_path}...")
    with open(role_prompts_path, 'r', encoding='utf-8') as f:
        role_prompts = json.load(f)

    eval_prompts = {}
    for role, data in role_prompts.items():
        if 'eval_prompt' in data:
            eval_prompts[role] = data['eval_prompt']

    print(f"Loaded eval prompts for {len(eval_prompts)} roles")
    return eval_prompts


def load_responses(responses_path: Path) -> pd.DataFrame:
    """Load responses from parquet file."""
    print(f"Loading responses from {responses_path}...")
    df = pd.read_parquet(responses_path)
    print(f"Loaded {len(df):,} responses")
    return df


def load_existing_evaluations(eval_path: Path) -> pd.DataFrame:
    """Load existing evaluations if they exist."""
    if eval_path.exists():
        print(f"Loading existing evaluations from {eval_path}...")
        df = pd.read_parquet(eval_path)
        print(f"Found {len(df):,} existing evaluations")
        return df
    else:
        print("No existing evaluations found, starting fresh")
        return pd.DataFrame()


def get_unprocessed_indices(responses_df: pd.DataFrame, eval_df: pd.DataFrame) -> List[int]:
    """Determine which response indices haven't been evaluated yet."""
    if eval_df.empty:
        return list(range(len(responses_df)))

    processed_indices = set(eval_df['response_idx'].values)
    all_indices = set(range(len(responses_df)))
    unprocessed = sorted(all_indices - processed_indices)

    print(f"Already evaluated: {len(processed_indices):,}")
    print(f"Remaining: {len(unprocessed):,}")

    return unprocessed


def construct_eval_prompt(eval_template: str, question: str, answer: str) -> str:
    """Construct the evaluation prompt by filling in the template."""
    prompt = eval_template.replace('{{question}}', question)
    prompt = prompt.replace('{{answer}}', answer)
    return prompt


def parse_score(response: str) -> Optional[int]:
    """Parse the score from the judge's response."""
    response = response.strip()

    # Try to extract a number 0-3
    match = re.search(r'^([0-3])$', response)
    if match:
        return int(match.group(1))

    # Try to find any digit 0-3 at the start
    match = re.search(r'^([0-3])', response)
    if match:
        return int(match.group(1))

    # Try to find any digit 0-3 anywhere
    match = re.search(r'([0-3])', response)
    if match:
        return int(match.group(1))

    return None


def evaluate_batch(
    llm: LLM,
    eval_prompts: List[str],
    indices: List[int],
    response_data: List[Dict],
    sampling_params: SamplingParams,
    judge_model: str
) -> List[Dict]:
    """Evaluate a batch of responses."""
    outputs = llm.generate(eval_prompts, sampling_params)

    results = []
    for idx, data, output in zip(indices, response_data, outputs):
        raw_response = output.outputs[0].text
        score = parse_score(raw_response)

        results.append({
            'response_idx': idx,
            'rollout_idx': data['rollout_idx'],
            'judge_model': judge_model,
            'role': data['role'],
            'role_category': data['role_category'],
            'instruction_idx': data['instruction_idx'],
            'question_idx': data['question_idx'],
            'score': score,
            'raw_judge_response': raw_response,
        })

    return results


def save_evaluations(evaluations: List[Dict], output_path: Path, mode='append'):
    """Save evaluations to parquet file."""
    new_df = pd.DataFrame(evaluations)

    if mode == 'append' and output_path.exists():
        existing_df = pd.read_parquet(output_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_parquet(output_path, compression='snappy', index=False)
    else:
        new_df.to_parquet(output_path, compression='snappy', index=False)

    print(f"Saved {len(evaluations)} evaluations to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model responses using LLM judge')
    parser.add_argument('--responses', type=str, required=True,
                        help='Input responses parquet file (e.g., responses_gemma-2-2b-it.parquet)')
    parser.add_argument('--judge-model', type=str, default='Qwen/Qwen2.5-14B-Instruct',
                        help='Judge model to use (default: Qwen/Qwen2.5-14B-Instruct)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference (default: 32)')
    parser.add_argument('--save-every', type=int, default=500,
                        help='Save evaluations every N responses (default: 500)')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                        help='Fraction of GPU memory to use (default: 0.9)')
    parser.add_argument('--max-model-len', type=int, default=4096,
                        help='Maximum model context length (default: 4096)')
    parser.add_argument('--quantization', type=str, default='bitsandbytes',
                        help='Quantization method: bitsandbytes (8-bit), awq, gptq, or none (default: bitsandbytes)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh, ignoring existing evaluations')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of responses to evaluate (for testing)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file name (default: evaluations_<responses_name>.parquet)')

    args = parser.parse_args()

    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    responses_path = data_dir / args.responses
    role_prompts_path = data_dir / "role_prompts.json"

    # Generate output filename if not specified
    if args.output:
        eval_path = data_dir / args.output
    else:
        responses_name = args.responses.replace('responses_', '').replace('.parquet', '')
        eval_path = data_dir / f"evaluations_{responses_name}.parquet"

    # Load data
    eval_prompts_by_role = load_role_prompts(role_prompts_path)
    responses_df = load_responses(responses_path)
    print(f"Output will be saved to: {eval_path}")

    # Determine what needs to be processed
    if args.no_resume:
        print("Starting fresh (--no-resume flag)")
        unprocessed_indices = list(range(len(responses_df)))
        if eval_path.exists():
            eval_path.unlink()
            print(f"Deleted existing {eval_path}")
    else:
        eval_df = load_existing_evaluations(eval_path)
        unprocessed_indices = get_unprocessed_indices(responses_df, eval_df)

    if not unprocessed_indices:
        print("All responses have been evaluated!")
        return

    # Apply limit if specified
    if args.limit:
        unprocessed_indices = unprocessed_indices[:args.limit]
        print(f"Limited to {args.limit} responses for testing")

    # Initialize vLLM
    print(f"\nInitializing vLLM with judge model: {args.judge_model}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"Max model length: {args.max_model_len}")
    print(f"Quantization: {args.quantization}")

    # Build vLLM arguments
    llm_kwargs = {
        "model": args.judge_model,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
    }

    # Add quantization if specified
    if args.quantization and args.quantization.lower() != 'none':
        llm_kwargs["quantization"] = args.quantization
        if args.quantization == 'bitsandbytes':
            llm_kwargs["load_format"] = "bitsandbytes"

    llm = LLM(**llm_kwargs)

    # Sampling parameters - we want deterministic output
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8,  # Only need a single digit
    )

    print(f"\nEvaluation parameters:")
    print(f"  Judge model: {args.judge_model}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Save every: {args.save_every} responses")

    # Process in batches
    all_evaluations = []
    batch_prompts = []
    batch_indices = []
    batch_data = []

    print(f"\nEvaluating {len(unprocessed_indices):,} responses...")

    start_time = time.time()
    parse_failures = 0

    with tqdm(total=len(unprocessed_indices), desc="Evaluating responses") as pbar:
        for i, idx in enumerate(unprocessed_indices):
            row = responses_df.iloc[idx]
            role = row['role']

            # Get the eval prompt template for this role
            if role == 'default_assistant':
                eval_template = DEFAULT_ASSISTANT_EVAL_PROMPT
            elif role in eval_prompts_by_role:
                eval_template = eval_prompts_by_role[role]
            else:
                # Skip if no eval prompt available
                print(f"\nWarning: No eval prompt for role '{role}', skipping")
                pbar.update(1)
                continue

            # Extract question from prompt (it's instruction + \n\n + question)
            prompt_parts = row['prompt'].split('\n\n', 1)
            question = prompt_parts[1] if len(prompt_parts) > 1 else row['prompt']

            # Construct eval prompt
            eval_prompt = construct_eval_prompt(eval_template, question, row['response'])

            batch_prompts.append(eval_prompt)
            batch_indices.append(idx)
            batch_data.append({
                'rollout_idx': row['rollout_idx'],
                'role': role,
                'role_category': row['role_category'],
                'instruction_idx': row['instruction_idx'],
                'question_idx': row['question_idx'],
            })

            # Process batch when full or at end
            if len(batch_prompts) >= args.batch_size or i == len(unprocessed_indices) - 1:
                batch_results = evaluate_batch(
                    llm, batch_prompts, batch_indices, batch_data,
                    sampling_params, args.judge_model
                )

                # Count parse failures
                for result in batch_results:
                    if result['score'] is None:
                        parse_failures += 1

                all_evaluations.extend(batch_results)
                pbar.update(len(batch_prompts))

                # Save periodically
                if len(all_evaluations) >= args.save_every or i == len(unprocessed_indices) - 1:
                    save_evaluations(all_evaluations, eval_path, mode='append')

                    elapsed = time.time() - start_time
                    processed = i + 1
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = len(unprocessed_indices) - processed
                    eta = remaining / rate if rate > 0 else 0

                    print(f"\nProgress: {processed:,}/{len(unprocessed_indices):,} "
                          f"({100*processed/len(unprocessed_indices):.1f}%)")
                    print(f"Rate: {rate:.1f} responses/sec")
                    print(f"ETA: {eta/60:.1f} minutes")
                    if parse_failures > 0:
                        print(f"Parse failures: {parse_failures}")

                    all_evaluations = []

                # Reset batch
                batch_prompts = []
                batch_indices = []
                batch_data = []

    # Final statistics
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total evaluated: {len(unprocessed_indices):,} responses")
    print(f"Average rate: {len(unprocessed_indices)/total_time:.1f} responses/sec")
    if parse_failures > 0:
        print(f"Total parse failures: {parse_failures} ({100*parse_failures/len(unprocessed_indices):.1f}%)")
    print(f"\nEvaluations saved to: {eval_path}")

    # Load and show score distribution
    final_df = pd.read_parquet(eval_path)
    print(f"\nScore distribution:")
    score_counts = final_df['score'].value_counts().sort_index()
    for score, count in score_counts.items():
        pct = 100 * count / len(final_df)
        label = {0: "Refused", 1: "Can't be role", 2: "AI with attributes", 3: "Fully playing role", None: "Parse failed"}
        print(f"  {score}: {count:,} ({pct:.1f}%) - {label.get(score, 'Unknown')}")


if __name__ == "__main__":
    main()
