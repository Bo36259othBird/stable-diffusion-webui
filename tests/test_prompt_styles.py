import os
import tempfile
import pytest
from modules.prompt_styles import PromptStyle, StyleDatabase, merge_prompts, apply_styles_to_prompt


class TestMergePrompts:
    def test_no_placeholder_appends(self):
        assert merge_prompts("beautiful landscape", "sunset") == "sunset, beautiful landscape"

    def test_placeholder_replaced(self):
        assert merge_prompts("a photo of {prompt}, 4k", "a cat") == "a photo of a cat, 4k"

    def test_empty_prompt(self):
        assert merge_prompts("style only", "") == "style only"

    def test_empty_style(self):
        assert merge_prompts("", "my prompt") == "my prompt"

    def test_both_empty(self):
        # edge case: both style and prompt are empty strings
        assert merge_prompts("", "") == ""


class TestApplyStylesToPrompt:
    def test_multiple_styles_applied(self):
        styles = [
            PromptStyle("s1", "vivid colors", ""),
            PromptStyle("s2", "ultra detailed", ""),
        ]
        result = apply_styles_to_prompt("portrait", styles)
        assert "portrait" in result
        assert "vivid colors" in result
        assert "ultra detailed" in result

    def test_no_styles_returns_original(self):
        assert apply_styles_to_prompt("portrait", []) == "portrait"


class TestStyleDatabase:
    def _make_db(self, rows: list) -> StyleDatabase:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8-sig") as f:
            f.write("name,prompt,negative_prompt\n")
            for row in rows:
                f.write(",".join(row) + "\n")
            path = f.name
        db = StyleDatabase(path)
        os.unlink(path)
        return db

    def test_load_styles(self):
        db = self._make_db([("cinematic", "cinematic lighting", "blurry")])
        assert "cinematic" in db.styles
        assert db.styles["cinematic"].prompt == "cinematic lighting"

    def test_apply_style(self):
        db = self._make_db([("vivid", "vivid colors", "dull")])
        result = db.apply_styles_to_prompt("portrait", ["vivid"])
        assert "vivid colors" in result

    def test_apply_negative_style(self):
        db = self._make_db([("vivid", "vivid colors", "dull, desaturated")])
        result = db.apply_negative_styles_to_prompt("", ["vivid"])
        assert "dull" in result

    def test_missing_style_ignored(self):
        db = self._make_db([])
        result = db.apply_styles_to_prompt("portrait", ["nonexistent"])
        assert result == "portrait"

    def test_add_and_save_load(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            db = StyleDatabase(path)
            db.add_style(PromptStyle("test", "test prompt", "test neg"))
            db.save()
            db2 = StyleDatabase(path)
            assert "test" in db2.styles
            assert db2.styles["test"].prompt == "test prompt"
        finally:
            os.unlink(path)

    def test_delete_style(self):
        db = self._make_db([("cinematic", "cinematic", "")])
        db.delete_style("cinematic")
        # verify the style is actually removed from the dict, not just falsy
        assert "cinematic" not in db.styles
