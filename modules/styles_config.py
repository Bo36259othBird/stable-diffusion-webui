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
    # Log skipped style names so I can catch typos in my styles.csv
    skipped = [name for name in style_names if name not in db.styles]
    if skipped:
        print(f"[styles_config] Warning: skipped unknown style(s): {skipped}")
    styled_prompt = db.apply_styles_to_prompt(prompt, valid_style_names)
    styled_negative = db.apply_negative_styles_to_prompt(negative_prompt, valid_style_names)
    return styled_prompt, styled_negative


def save_style(name: str, prompt: str, negative_prompt: str = "", path: str = DEFAULT_STYLES_FILE):
    """Add or update a style and persist to disk.

    Note: existing style with the same name will be overwritten without confirmation.
    """
    db = get_style_database(path)
    if name in db.styles:
        print(f"[styles_config] Overwriting existing style: '{name}'")
    db.add_style(PromptStyle(name=name, prompt=prompt, negative_prompt=negative_prompt))
    db.save()
    # Reload the in-memory db so callers immediately see the updated style
    # without needing to manually call reload_styles(). Noticed this was a
    # footgun when testing new styles interactively.
    _db = db


def delete_style(name: str, path: str = DEFAULT_STYLES_FILE):
    """Remove a style by name and persist to disk."""
    db = get_style_database(path)
    if name not in db.styles:
        print(f"[styles_config] Warning: tried to delete non-existent style: '{name}'")
        return
    db.delete_style(name)
    db.save()
