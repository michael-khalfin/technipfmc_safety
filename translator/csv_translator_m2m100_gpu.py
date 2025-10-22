#!/usr/bin/env python3
"""
Translate specific columns in a CSV file into English using the M2M100 model.
Supports automatic source language detection and GPU acceleration.
"""

import argparse
import pandas as pd
import langid
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


def clean_text(x: str) -> str:
    """Remove line breaks and extra spaces from text."""
    return str(x).replace("\n", " ").replace("\r", " ").strip()


def normalize_src_lang(lang: str, tokenizer) -> str:
    """Normalize detected language codes to match M2M100 expected codes."""
    alias = {
        "no": "nb",       # Norwegian Bokmål
        "iw": "he",       # Hebrew (legacy code)
        "zh-cn": "zh",    # Simplified Chinese
        "zh-tw": "zh",    # Traditional Chinese
    }
    lang = alias.get(lang, lang)
    return lang if lang in tokenizer.lang_code_to_id else ""


def translate_text(text, src_lang, tgt_lang, tokenizer, model, device):
    """Translate a single text string from src_lang to tgt_lang."""
    try:
        tokenizer.src_lang = src_lang
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        gen = model.generate(**enc, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
        return tokenizer.decode(gen[0], skip_special_tokens=True)
    except Exception:
        return text


def main():
    parser = argparse.ArgumentParser(description="Translate CSV columns into English using M2M100.")
    parser.add_argument("--csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--columns", nargs="+", required=True, help="Column names to translate.")
    parser.add_argument("--out", required=True, help="Path to save the output CSV file.")
    parser.add_argument("--model", default="models/m2m100_418M", help="Model name or local path.")
    parser.add_argument("--target", default="en", help="Target language (default: English).")
    args = parser.parse_args()

    # --- GPU auto-detection ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    tokenizer = M2M100Tokenizer.from_pretrained(args.model)
    model = M2M100ForConditionalGeneration.from_pretrained(args.model)
    model.to(device)
    model.eval()

    df = pd.read_csv(args.csv)

    # --- Translation loop ---
    with torch.inference_mode():
        for col in args.columns:
            if col not in df.columns:
                print(f"[WARN] Column '{col}' not found, skipping.")
                continue

            langs, new_texts = [], []
            for val in df[col]:
                if pd.isna(val) or not str(val).strip():
                    langs.append(None)
                    new_texts.append(val)
                    continue

                text = clean_text(val)
                lang, _ = langid.classify(text)
                langs.append(lang)

                if lang == args.target:
                    new_texts.append(text)
                else:
                    src = normalize_src_lang(lang, tokenizer)
                    new_texts.append(
                        translate_text(text, src, args.target, tokenizer, model, device) if src else text
                    )

            # Add new columns for detected language and translated text
            df[f"{col}_lang"] = langs
            df[f"{col}_{args.target}"] = new_texts

    # --- Keep only relevant columns ---
    keep_cols = []
    for col in args.columns:
        keep_cols.extend([f"{col}_lang", f"{col}_{args.target}"])

    df_out = df[keep_cols]
    df_out.to_csv(args.out, index=False)
    print(f"✅ Done. Translated columns saved to {args.out}")


if __name__ == "__main__":
    main()
