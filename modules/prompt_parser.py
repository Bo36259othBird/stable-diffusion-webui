"""Utility functions for parsing and processing prompts in stable-diffusion-webui."""

import re
from typing import Optional


PROMPT_ATTENTION_RE = re.compile(
    r"""
    \\\(|  # escaped opening paren
    \\\)|  # escaped closing paren
    \\\[|  # escaped opening bracket
    \\\]|  # escaped closing bracket
    \\\\|  # escaped backslash
    \\|    # lone backslash
    \(|    # opening paren
    \[|    # opening bracket
    :\s*([+-]?[.\d]+)\s*\)|  # :number)
    \)|    # closing paren
    \]|    # closing bracket
    [^\\()\[\]]+|  # anything else
    .
    """,
    re.X,
)


def parse_prompt_attention(text: str) -> list[list]:
    """Parse prompt attention weights from text.

    Returns a list of [text, weight] pairs.
    Parentheses increase weight by 1.1x, brackets decrease by ~0.909x.
    """
    result = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position: int, multiplier: float) -> None:
        for p in range(start_position, len(result)):
            result[p][1] *= multiplier

    for m in PROMPT_ATTENTION_RE.finditer(text):
        token = m.group(0)
        weight = m.group(1)

        if token.startswith("\\\\"):
            result.append([token[1:], 1.0])
        elif token == "(":
            round_brackets.append(len(result))
        elif token == "[":
            square_brackets.append(len(result))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif token == ")" and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif token == "]" and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(r"\\(.)", token)
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    if part:
                        result.append([part, 1.0])
                else:
                    result.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)
    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if not result:
        result.append(["", 1.0])

    # Merge consecutive entries with the same weight
    i = 0
    while i + 1 < len(result):
        if result[i][1] == result[i + 1][1]:
            result[i][0] += result[i + 1][0]
            result.pop(i + 1)
        else:
            i += 1

    return result


def get_learned_conditioning_prompt_schedules(
    prompts: list[str], steps: int
) -> list[list]:
    """Build per-step prompt schedules from a list of prompt strings."""
    schedules = []
    for prompt in prompts:
        schedule = [[steps, parse_prompt_attention(prompt)]]
        schedules.append(schedule)
    return schedules
