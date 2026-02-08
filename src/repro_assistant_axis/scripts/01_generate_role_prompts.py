import json
import os
from pathlib import Path
from typing import Dict, List, Any
import anthropic
from tqdm import tqdm
import time
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROMPTS_DIR = BASE_DIR / "prompts"
ROLES_FILE = DATA_DIR / "roles.json"
OUTPUT_FILE = DATA_DIR / "role_prompts.json"
TEMPLATE_FILE = PROMPTS_DIR / "role_generation_template.txt"


def load_prompt_template() -> str:
    with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
        return f.read()


def load_roles() -> List[Dict[str, str]]:
    with open(ROLES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_role_data(client: anthropic.Anthropic, role: str, prompt_template: str, model: str = "claude-sonnet-4-20250514") -> Dict[str, Any]:
    prompt = prompt_template.format(ROLE=role)

    message = client.messages.create(
        model=model,
        max_tokens=8000,
        temperature=1.0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    response_text = message.content[0].text

    try:
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        json_str = response_text[start_idx:end_idx]
        data = json.loads(json_str)

        assert "instruction" in data, "Missing 'instruction' field"
        assert "questions" in data, "Missing 'questions' field"
        assert "eval_prompt" in data, "Missing 'eval_prompt' field"
        assert len(data["instruction"]) == 5, f"Expected 5 instructions, got {len(data['instruction'])}"
        assert len(data["questions"]) == 40, f"Expected 40 questions, got {len(data['questions'])}"

        return data
    except (json.JSONDecodeError, AssertionError) as e:
        print(f"Error parsing response for role '{role}': {e}")
        print(f"Response: {response_text[:500]}...")
        raise


def generate_all_role_prompts(
    api_key: str | None = None,
    start_idx: int = 0,
    end_idx: int | None = None,
    model: str = "claude-sonnet-4-20250514",
    resume: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Generate prompts for all roles.

    Args:
        start_idx: Starting index in roles list (for resuming)
        end_idx: Ending index in roles list (None = all roles)
        resume: If True, load existing results and skip already processed roles

    Returns:
        Dictionary mapping role names to their generated data
    """
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    template = load_prompt_template()

    roles = load_roles()
    if end_idx is None:
        end_idx = len(roles)

    roles_to_process = roles[start_idx:end_idx]

    # Load existing results if resuming
    results = {}
    if resume and OUTPUT_FILE.exists():
        print(f"Loading existing results from {OUTPUT_FILE}")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing role prompts")

    print(f"Processing roles {start_idx} to {end_idx} ({len(roles_to_process)} roles)")

    for i, role_data in enumerate(tqdm(roles_to_process, desc="Generating role prompts")):
        role_name = role_data["role"]

        if role_name in results:
            print(f"Skipping {role_name} (already processed)")
            continue

        try:
            role_prompts = generate_role_data(client, role_name, template, model=model)

            results[role_name] = {
                "category": role_data["category"],
                "instruction": role_prompts["instruction"],
                "questions": role_prompts["questions"],
                "eval_prompt": role_prompts["eval_prompt"]
            }

            if i % 5 == 0 or i == len(roles_to_process) - 1:
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)

            # Rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"Error processing role '{role_name}': {e}")
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            raise

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nGeneration complete! Processed {len(results)} roles.")
    print(f"Results saved to {OUTPUT_FILE}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate role prompts for Assistant Axis replication")
    parser.add_argument("--start", type=int, default=0, help="Starting role index")
    parser.add_argument("--end", type=int, default=None, help="Ending role index (None = all)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514", help="Claude model to use")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing results")
    parser.add_argument("--api-key", type=str, default=None, help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")

    args = parser.parse_args()

    generate_all_role_prompts(
        api_key=args.api_key,
        start_idx=args.start,
        end_idx=args.end,
        model=args.model,
        resume=not args.no_resume
    )
