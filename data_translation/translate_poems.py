from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from prompts import SYSTEM_PROMPT, build_user_prompt


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_api_key() -> str:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""


def translate_text(
    client,
    model_name: str,
    config,
    text: str,
    max_retries: int,
    sleep_seconds: float,
) -> str:
    if not text:
        return ""
    prompt = build_user_prompt(text)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            out = (response.text or "").strip()
            return out
        except Exception as exc:
            last_err = exc
            if attempt < max_retries - 1:
                time.sleep(sleep_seconds)
    raise RuntimeError(f"translation failed after {max_retries} attempts: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate poems to modern Azerbaijani using Gemini API.")
    parser.add_argument("--input", type=str, default="poems_cleaned.parquet")
    parser.add_argument("--output", type=str, default="data_translation/poems_translated.parquet")
    parser.add_argument("--model", type=str, default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--sleep", type=float, default=float(os.getenv("GEMINI_SLEEP_SECONDS", "2.0")))
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Translate only first N rows (0 = all)")
    parser.add_argument("--start", type=int, default=0, help="Start row index")
    parser.add_argument("--out-col", type=str, default="modern_text")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    load_env_file(root_dir / ".env")

    api_key = get_api_key()
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY or GOOGLE_API_KEY in .env")

    try:
        from google import genai
        from google.genai import types
    except Exception as exc:  # pragma: no cover - import guard
        raise SystemExit("google-genai is not installed. Install it first: pip install google-genai") from exc

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)

    input_path = root_dir / args.input
    output_path = root_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    if args.limit > 0:
        df = df.iloc[: args.limit]

    if output_path.exists():
        df_out = pd.read_parquet(output_path)
        if len(df_out) != len(df):
            raise SystemExit("Output row count does not match input. Remove output or use matching input.")
    else:
        df_out = df.copy()
        if args.out_col not in df_out.columns:
            df_out[args.out_col] = ""

    error_log_path = output_path.with_suffix(".errors.jsonl")

    translated = 0
    for idx in range(args.start, len(df_out)):
        current = df_out.at[idx, args.out_col]
        if isinstance(current, str) and current.strip():
            continue

        source_text = df_out.at[idx, "text"] if "text" in df_out.columns else ""
        try:
            translated_text = translate_text(
                client=client,
                model_name=args.model,
                config=config,
                text=source_text,
                max_retries=args.max_retries,
                sleep_seconds=args.sleep,
            )
            df_out.at[idx, args.out_col] = translated_text
            translated += 1
            print(f"Translated {translated}/{len(df_out)}", flush=True)
        except Exception as exc:
            payload = {
                "row": int(idx),
                "error": str(exc),
            }
            with error_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        if translated and translated % args.save_every == 0:
            df_out.to_parquet(output_path, index=False)
            print(f"Translated {translated}/{len(df_out)}", flush=True)

        if args.sleep > 0:
            time.sleep(args.sleep)

    df_out.to_parquet(output_path, index=False)
    print(f"Translated {translated}/{len(df_out)}", flush=True)


if __name__ == "__main__":
    main()
