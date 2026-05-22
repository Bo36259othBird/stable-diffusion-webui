"""Tests for modules/prompt_parser.py"""

import pytest
from modules.prompt_parser import parse_prompt_attention, get_learned_conditioning_prompt_schedules


class TestParsePromptAttention:
    def test_plain_text_has_weight_one(self):
        result = parse_prompt_attention("a cat")
        assert result == [["a cat", 1.0]]

    def test_round_brackets_increase_weight(self):
        result = parse_prompt_attention("(a cat)")
        assert len(result) == 1
        assert result[0][0] == "a cat"
        assert abs(result[0][1] - 1.1) < 1e-6

    def test_square_brackets_decrease_weight(self):
        result = parse_prompt_attention("[a cat]")
        assert len(result) == 1
        assert result[0][0] == "a cat"
        assert abs(result[0][1] - (1 / 1.1)) < 1e-6

    def test_explicit_weight(self):
        result = parse_prompt_attention("(a cat:1.5)")
        assert len(result) == 1
        assert result[0][0] == "a cat"
        assert abs(result[0][1] - 1.5) < 1e-6

    def test_nested_brackets(self):
        result = parse_prompt_attention("((a cat))")
        assert len(result) == 1
        assert result[0][0] == "a cat"
        assert abs(result[0][1] - 1.1 * 1.1) < 1e-6

    def test_empty_string_returns_empty_weight(self):
        result = parse_prompt_attention("")
        assert result == [["", 1.0]]

    def test_multiple_segments(self):
        result = parse_prompt_attention("a (cat) and a dog")
        texts = [r[0] for r in result]
        assert "cat" in texts
        weights = {r[0]: r[1] for r in result}
        assert abs(weights["cat"] - 1.1) < 1e-6
        assert abs(weights.get("a ", weights.get(" and a dog", 1.0)) - 1.0) < 1e-6

    def test_unmatched_round_bracket_applies_default_multiplier(self):
        result = parse_prompt_attention("(a cat")
        assert len(result) == 1
        assert result[0][0] == "a cat"
        assert abs(result[0][1] - 1.1) < 1e-6

    def test_unmatched_square_bracket_applies_default_multiplier(self):
        result = parse_prompt_attention("[a cat")
        assert len(result) == 1
        assert result[0][0] == "a cat"
        assert abs(result[0][1] - (1 / 1.1)) < 1e-6

    def test_consecutive_same_weight_merged(self):
        result = parse_prompt_attention("hello world")
        assert len(result) == 1
        assert result[0][0] == "hello world"


class TestGetLearnedConditioningPromptSchedules:
    def test_single_prompt_schedule(self):
        schedules = get_learned_conditioning_prompt_schedules(["a cat"], steps=20)
        assert len(schedules) == 1
        assert schedules[0][0][0] == 20
        assert schedules[0][0][1] == [["a cat", 1.0]]

    def test_multiple_prompts(self):
        prompts = ["a cat", "a dog"]
        schedules = get_learned_conditioning_prompt_schedules(prompts, steps=30)
        assert len(schedules) == 2
        assert schedules[0][0][0] == 30
        assert schedules[1][0][0] == 30

    def test_empty_prompts_list(self):
        schedules = get_learned_conditioning_prompt_schedules([], steps=10)
        assert schedules == []
