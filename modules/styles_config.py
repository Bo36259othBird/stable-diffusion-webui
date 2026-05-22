import os
from modules.prompt_styles import StyleDatabase, PromptStyle

DEFAULT_STYLES_FILE = os.environ.get("STYLES_CSV", "styles.csv")

_db: StyleDatabase | None = None


def get_style_database(path: str = DEFAULT_STYLES_FILE) -> StyleDatabase:
    global _db
    if _db is None or _db.path != path:
        _db = StyleDatabase(path)
    return _db


def reload_styles(path: str = DEFAULT_STYLES_FILE) -> StyleDatabase:
    global _db
    _db = StyleDatabase(path)
    return _db


def list_style_names(path: str = DEFAULT_STYLES_FILE) -> list:
    db = get_style_database(path)
    return list(db.styles.keys())


def apply_styles(prompt: str, negative_prompt: str, style_names: list, path: str = DEFAULT_STYLES_FILE):
    """Apply named styles to both positive and negative prompts.

    Returns a tuple (styled_prompt, styled_negative_prompt).
    Silently skips any style names that don't exist in the database.
    """
    db = get_style_database(path)
    # Filter out style names that aren't in the database to avoid errors
    valid_style_names = [name for name in style_names if name in db.styles]
    styled_prompt = db.apply_styles_to_prompt(prompt, valid_style_names)
    styled_negative = db.apply_negative_styles_to_prompt(negative_prompt, valid_style_names)
    return styled_prompt, styled_negative


def save_style(name: str, prompt: str, negative_prompt: str = "", path: str = DEFAULT_STYLES_FILE):
    """Add or update a style and persist to disk."""
    db = get_style_database(path)
    db.add_style(PromptStyle(name=name, prompt=prompt, negative_prompt=negative_prompt))
    db.save()


def delete_style(name: str, path: str = DEFAULT_STYLES_FILE):
    """Remove a style by name and persist to disk."""
    db = get_style_database(path)
    db.delete_style(name)
    db.save()
