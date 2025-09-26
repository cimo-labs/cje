"""Create test data for CJE pipeline testing."""

import json
import numpy as np
from pathlib import Path


def create_test_data() -> None:
    """Create various test datasets for different scenarios."""

    data_dir = Path(__file__).parent

    # 1. Basic test data with all required fields
    basic_data = []
    np.random.seed(42)

    for i in range(20):
        # Create realistic log probabilities
        base_logprob = -30 - np.random.exponential(10)

        record = {
            "prompt": f"Test prompt {i}",
            "response": f"Test response {i} with some content",
            "reward": 0.5
            + 0.3 * np.sin(i / 3)
            + 0.1 * np.random.randn(),  # Varying rewards
            "base_policy_logprob": base_logprob,
            "target_policy_logprobs": {
                "pi_improved": base_logprob
                + np.random.normal(2, 5),  # Sometimes better
                "pi_worse": base_logprob - np.random.exponential(5),  # Usually worse
                "pi_similar": base_logprob + np.random.normal(0, 2),  # Similar to base
            },
        }
        basic_data.append(record)

    # Save basic data
    with open(data_dir / "basic_test_data.jsonl", "w") as f:
        for record in basic_data:
            f.write(json.dumps(record) + "\n")

    # 2. Data with missing values (failed API calls)
    missing_data = []
    for i in range(15):
        base_logprob = -35 - np.random.exponential(8)

        record = {
            "prompt": f"Test prompt {i}",
            "response": f"Test response {i}",
            "reward": 0.6 + 0.2 * np.random.randn(),
            "base_policy_logprob": base_logprob if i % 3 != 0 else None,  # Some missing
            "target_policy_logprobs": {
                "pi_a": base_logprob + np.random.normal(1, 3) if i % 4 != 0 else None,
                "pi_b": base_logprob + np.random.normal(-1, 3) if i % 5 != 0 else None,
            },
        }
        missing_data.append(record)

    with open(data_dir / "missing_values_data.jsonl", "w") as f:
        for record in missing_data:
            f.write(json.dumps(record) + "\n")

    # 3. Data with extreme weights (edge cases)
    extreme_data = []
    for i in range(10):
        if i < 5:
            # Normal cases
            base_logprob = -25 - np.random.exponential(5)
            target_adjustment = np.random.normal(0, 3)
        else:
            # Extreme cases
            base_logprob = -100 - np.random.exponential(20)
            target_adjustment = np.random.choice([50, -50])  # Very different

        record = {
            "prompt": f"Extreme test {i}",
            "response": f"Response {i}",
            "reward": np.clip(0.5 + 0.5 * np.random.randn(), 0, 1),
            "base_policy_logprob": base_logprob,
            "target_policy_logprobs": {
                "pi_extreme": base_logprob + target_adjustment,
            },
        }
        extreme_data.append(record)

    with open(data_dir / "extreme_weights_data.jsonl", "w") as f:
        for record in extreme_data:
            f.write(json.dumps(record) + "\n")

    # 4. Data for judge calibration testing (with oracle labels)
    judge_data = []

    # True relationship: oracle = 0.1 + 0.8 * sigmoid(judge - 5) + noise
    for i in range(30):
        judge_score = np.random.uniform(0, 10)
        oracle_label = (
            0.1 + 0.8 / (1 + np.exp(-(judge_score - 5))) + 0.05 * np.random.randn()
        )
        oracle_label = np.clip(oracle_label, 0, 1)

        record = {
            "prompt": f"Judge test {i}",
            "response": f"Response to judge {i}",
            "judge_score": judge_score,
            "oracle_label": (
                oracle_label if i < 10 else None
            ),  # Only first 10 have oracle
            "base_policy_logprob": -30 - np.random.exponential(10),
            "target_policy_logprobs": {
                "pi_test": -28 - np.random.exponential(10),
            },
        }
        judge_data.append(record)

    with open(data_dir / "judge_calibration_data.jsonl", "w") as f:
        for record in judge_data:
            f.write(json.dumps(record) + "\n")

    # 5. Chat format data (for teacher forcing tests)
    chat_data = []
    for i in range(5):
        record = {
            "chat": [
                {"role": "user", "content": f"Question {i}: What is {i} + {i}?"},
                {"role": "assistant", "content": f"The answer is {2*i}."},
            ],
            "reward": 0.8 + 0.1 * np.random.randn(),
            "model": "test-model",
        }
        chat_data.append(record)

    with open(data_dir / "chat_data.jsonl", "w") as f:
        for record in chat_data:
            f.write(json.dumps(record) + "\n")

    print("Test data created:")
    print(f"  - basic_test_data.jsonl: {len(basic_data)} samples")
    print(f"  - missing_values_data.jsonl: {len(missing_data)} samples")
    print(f"  - extreme_weights_data.jsonl: {len(extreme_data)} samples")
    print(f"  - judge_calibration_data.jsonl: {len(judge_data)} samples")
    print(f"  - chat_data.jsonl: {len(chat_data)} samples")


if __name__ == "__main__":
    create_test_data()
