import pandas as pd
import requests
from tqdm import tqdm
import time


# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"  
DESC_COL = "DESCRIPTION"


def translate_text(text: str, source_lang: str = "auto", target_lang: str = "English", retries: int = 3) -> str:
    """
    Calls an Ollama LLM to translate the given text.
    """
    prompt = f"Translate the following text from {source_lang} to {target_lang}:\n\n{text}\n\nTranslation:"
    for attempt in range(retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            print(f"[Retry {attempt+1}] Error translating: {e}")
            time.sleep(2)
    return "[TRANSLATION_FAILED]"


def translate_dataframe(df: pd.DataFrame, text_column: str, batch_size: int = 1) -> pd.DataFrame:
    """
    Translates each row's text in a given DataFrame column using Ollama.
    """
    translated_texts = []
    for i in tqdm(range(0, len(df), batch_size), desc="Translating"):
        batch = df.iloc[i:i+batch_size]
        for text in batch[text_column]:
            translated = translate_text(text)
            translated_texts.append(translated)
    df[DESC_COL] = translated_texts
    return df



if __name__ == "__main__":
    df = pd.read_csv("data/combined.csv")
    translated_df = translate_dataframe(df, DESC_COL)
    translated_df.to_csv("data/combined_translated.csv", index = False)
