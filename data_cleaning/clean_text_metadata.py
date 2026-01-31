from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


def remove_leading_template(text: str) -> tuple[str, bool]:
    # Remove a leading template block like:
    # {{...\n| ...\n| ...\n}}\n
    # or:
    # | ...\n| ...\n}}\n
    lines = text.splitlines()
    if not lines:
        return text, False

    i = 0
    had_open = False
    if lines[i].lstrip().startswith("{{"):
        had_open = True
        i += 1

    start_i = i
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        i += 1

    # Remove if we saw at least one metadata line and then a closing "}}" line.
    if i > start_i and i < len(lines) and lines[i].strip().startswith("}}"):
        i += 1
        return "\n".join(lines[i:]), True

    # If we saw an opening "{{" and then immediately "}}", remove that too.
    if had_open and i < len(lines) and lines[i].strip().startswith("}}"):
        i += 1
        return "\n".join(lines[i:]), True

    return text, False


def remove_table_of_contents(text: str) -> tuple[str, bool]:
    lines = text.splitlines()
    non_empty = [line for line in lines if line.strip()]
    if len(non_empty) < 5:
        return text, False

    bullet_count = sum(1 for line in non_empty if line.lstrip().startswith("*"))
    bullet_ratio = bullet_count / len(non_empty)

    # If the text is mostly bullet items, it's likely a TOC section, not a poem.
    if bullet_count >= 5 and bullet_ratio >= 0.6:
        remaining = [line for line in lines if line.strip() and not line.lstrip().startswith("*")]
        if not remaining:
            return "", True

        # If only short headings remain, drop them too.
        total_words = sum(len(line.strip().split()) for line in remaining)
        if all(len(line.strip()) <= 40 for line in remaining) and total_words <= 6:
            return "", True

        return "\n".join(remaining).strip("\n"), True

    return text, False


def clean_text(text: str) -> tuple[str, bool, bool]:
    if not isinstance(text, str):
        return "", False, False

    # Normalize line endings and trim leading whitespace.
    s = text.replace("\r\n", "\n").replace("\r", "\n").lstrip()
    s2, template_removed = remove_leading_template(s)
    # Trim leading blank lines after template removal.
    s2 = s2.lstrip("\n")
    s3, toc_removed = remove_table_of_contents(s2)
    return s3, template_removed, toc_removed


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    data_path = root_dir / "poems.parquet"
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)
    texts = df["text"].fillna("")

    cleaned = []
    template_flags = []
    toc_flags = []
    changed = 0

    for text in texts:
        cleaned_text, template_removed, toc_removed = clean_text(text)
        cleaned.append(cleaned_text)
        template_flags.append(template_removed)
        toc_flags.append(toc_removed)
        if cleaned_text != text:
            changed += 1

    df_clean = df.copy()
    df_clean["text"] = cleaned

    out_path = out_dir / "poems_cleaned.parquet"
    df_clean.to_parquet(out_path, index=False)

    report = {
        "dataset": str(data_path.name),
        "rows": int(len(df)),
        "changed_rows": int(changed),
        "template_removed_rows": int(sum(1 for f in template_flags if f)),
        "toc_removed_rows": int(sum(1 for f in toc_flags if f)),
        "empty_after_cleaning": int(sum(1 for t in cleaned if not t.strip())),
    }

    report_path = out_dir / "cleaning_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
