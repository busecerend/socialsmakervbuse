import os
import json
import csv
import base64
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any

from PIL import Image, UnidentifiedImageError
import torch
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import openai

# ---------------- Configuration ----------------

openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-4o"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "image_inputs")
CSV_FILE = os.path.join(PROJECT_ROOT, "data", "csv", "instameme_analysis.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "json", "meme_metadata.json")

GPT_CALL_THRESHOLD = 30

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# ---------------- Model Initialization ----------------

logging.info("Loading models...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
logging.info("Models loaded.")


# ---------------- Utility Functions ----------------

def extract_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def read_existing() -> Dict[str, Any]:
    if not os.path.exists(OUTPUT_FILE):
        return {}
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["header"]: item for item in data}


def read_csv() -> List[Dict[str, str]]:
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f)]


def base64_image(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        logging.warning(f"Image file not found: {path}")
        return None
    except Exception as e:
        logging.error(f"Error reading image {path}: {e}")
        return None


def embed_text(text: str) -> List[float]:
    return sentence_model.encode(text).tolist()


def embed_image(path: str) -> Optional[List[float]]:
    try:
        img = Image.open(path).convert("RGB")
        inputs = clip_processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
        return feats[0].cpu().tolist()
    except FileNotFoundError:
        logging.warning(f"Image file not found for embedding: {path}")
        return None
    except UnidentifiedImageError:
        logging.error(f"Cannot identify image file: {path}")
        return None
    except Exception as e:
        logging.error(f"Error embedding image {path}: {e}")
        return None


def estimate_viral_score(header: str, texts: List[str]) -> int:
    score = abs(hash(header + "".join(texts))) % 100
    return score


def build_prompt(header: str, texts: List[str], joke: str, score: int) -> List[Dict[str, str]]:
    prompt_text = f"""
You are a meme analyst in music culture tasked with generating detailed JSON metadata analysis.

Header: {header}
Texts: {texts}
Joke: {joke}

Return JSON with fields:
- emotion (string)
- theme_tags (list of strings)
- style (string)
- placement_hint (string)
- caption_tone (string)
- cultural_reference (string)
- visual_summary (string)
- viral_score (int)
- viral_reasoning (string)
- caption_suggestion (string)
"""
    return [
        {"role": "system", "content": "You are a meme analyst in music culture. Provide detailed JSON metadata."},
        {"role": "user", "content": prompt_text}
    ]


def gpt_analyze(header: str, texts: List[str], joke: str, score: int) -> Dict[str, Any]:
    if score < GPT_CALL_THRESHOLD:
        logging.info(f"Skipping GPT call for meme '{header}' due to low viral score ({score})")
        return {}

    try:
        prompt = build_prompt(header, texts, joke, score)
        max_tokens = 400 if score < 33 else 600 if score < 66 else 800
        response = openai.chat.completions.create(
            model=MODEL,
            messages=prompt,
            temperature=0.5,
            max_tokens=max_tokens
        )
        content = response.choices[0].message.content
        json_str = extract_json(content)
        return json.loads(json_str)
    except Exception as e:
        logging.error(f"GPT analysis error for meme '{header}': {e}")
        return {}


# ---------------- Main Processing ----------------

def process_meme(meme: Dict[str, str]) -> Optional[Dict[str, Any]]:
    header = meme.get("header", "").strip()
    if not header:
        logging.warning("Skipping meme with missing header")
        return None

    texts = [meme.get(f"text{i}", "").strip() for i in range(1, 5)]
    joke = meme.get("joke", "").strip()

    viral_score = estimate_viral_score(header, texts)

    image_data = []
    for i in range(1, 5):
        img_col = f"img{i}"
        img_name = meme.get(img_col, "").strip()
        if not img_name:
            img_name = f"{i:03d}.png"

        img_path = os.path.join(IMAGE_DIR, img_name)
        img_b64 = base64_image(img_path)  # <-- burada base64 kodunu alıyoruz
        img_emb = embed_image(img_path)

        image_data.append({
            "text": texts[i-1],
            "image": img_name,
            "image_base64": img_b64,    # <-- base64 burada
            "embedding": img_emb,
        })

    text_embedding = embed_text(header + " " + joke)

    analysis = gpt_analyze(header, texts, joke, viral_score)

    return {
        "header": header,
        "joke": joke,
        "text_embedding": text_embedding,
        "texts": image_data,
        "analysis": analysis,
        "viral_score": viral_score,
    }


def main():
    logging.info("Starting meme analysis process...")

    memes = read_csv()
    existing = read_existing()

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for meme in memes:
            header = meme.get("header", "").strip()
            if not header:
                logging.warning("Skipping meme with missing header")
                continue
            if header in existing:
                logging.info(f"Skipping already processed meme: {header}")
                continue
            futures.append(executor.submit(process_meme, meme))

        for future in as_completed(futures):
            result = future.result()
            if result:
                existing[result["header"]] = result
                logging.info(f"Processed meme: {result['header']}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(list(existing.values()), f, indent=2, ensure_ascii=False)

    logging.info(f"Analysis complete. Processed {len(existing)} memes total.")


if __name__ == "__main__":
    main()
