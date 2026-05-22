import csv
import os
from collections import namedtuple

PromptStyle = namedtuple("PromptStyle", ["name", "prompt", "negative_prompt"])


def merge_prompts(style_prompt: str, prompt: str) -> str:
    if "{prompt}" in style_prompt:
        return style_prompt.replace("{prompt}", prompt)
    elif prompt:
        return f"{prompt}, {style_prompt}" if style_prompt else prompt
    return style_prompt


def apply_styles_to_prompt(prompt: str, styles: list) -> str:
    for style in styles:
        prompt = merge_prompts(style.prompt, prompt)
    return prompt


class StyleDatabase:
    def __init__(self, path: str):
        self.path = path
        self.styles: dict[str, PromptStyle] = {}
        self.load()

    def load(self):
        self.styles = {}
        if not os.path.exists(self.path):
            return
        with open(self.path, newline="", encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row.get("name", "").strip()
                if not name:
                    continue
                self.styles[name] = PromptStyle(
                    name=name,
                    prompt=row.get("prompt", ""),
                    negative_prompt=row.get("negative_prompt", ""),
                )

    def save(self):
        # Ensure parent directory exists before writing; skip makedirs if path has no directory component
        if os.path.dirname(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["name", "prompt", "negative_prompt"])
            writer.writeheader()
            # Sort styles alphabetically by name for easier manual editing of the CSV
            for style in sorted(self.styles.values(), key=lambda s: s.name.lower()):
                writer.writerow({"name": style.name, "prompt": style.prompt, "negative_prompt": style.negative_prompt})

    def apply_styles_to_prompt(self, prompt: str, style_names: list) -> str:
        styles = [self.styles[name] for name in style_names if name in self.styles]
        return apply_styles_to_prompt(prompt, styles)

    def apply_negative_styles_to_prompt(self, prompt: str, style_names: list) -> str:
        styles = [self.styles[name] for name in style_names if name in self.styles]
        negative_styles = [PromptStyle(s.name, s.negative_prompt, "") for s in styles]
        return apply_styles_to_prompt(prompt, negative_styles)

    def add_style(self, style: PromptStyle):
        self.styles[style.name] = style

    def delete_style(self, name: str):
        self.styles.pop(name, None)
