import csv
import os
import random
from typing import List, Dict, Optional
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

CSV_FILE = "data/csv/instameme_analysis.csv"
MODEL_NAME = "gpt-4o"


def read_existing_memes() -> List[Dict[str, str]]:
    memes = []
    if not os.path.exists(CSV_FILE):
        return memes
    with open(CSV_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            memes.append(row)
    return memes


def create_style_prompt(existing_memes: List[Dict[str, str]]) -> str:
    examples = random.sample(existing_memes, min(8, len(existing_memes)))
    prompt = """You are a music producer meme creator. Based on these examples, create NEW music producer memes that follow the exact same format and humor style:

EXAMPLES:
"""
    for i, meme in enumerate(examples, 1):
        prompt += f"""
Example {i}:
Header: {meme['header']}
Text1: {meme['text1']} | Text2: {meme['text2']}
Text3: {meme['text3']} | Text4: {meme['text4']}
Joke: {meme['joke']}
Image descriptions: {meme['img1']} | {meme['img2']} | {meme['img3']} | {meme['img4']}
"""

    prompt += """
STYLE NOTES:
- Headers are relatable music producer situations/scenarios
- Text1-4 are short, punchy phrases that fit a 2x2 grid (read left-to-right, top-to-bottom)
- Jokes explain the humor in one sentence
- Image descriptions are brief but specific
- Humor focuses on music production struggles, habits, quirks, and inside jokes
- Use music production terminology (plugins, DAWs, mixing, etc.)
- Keep it authentic to producer culture

⚠️ Generate 5 COMPLETELY NEW meme concepts.
⚠️ Return ONLY the memes in this exact CSV format (no newlines, no extra text):

header,text1,text2,text3,text4,joke,img1,img2,img3,img4

⚠️ Do NOT include headers, do NOT use quotes, do NOT write explanations.
Only return 5 lines of CSV data. No extra line breaks.
"""
    return prompt


def generate_meme_candidates(existing_memes: List[Dict[str, str]]) -> List[str]:
    prompt = create_style_prompt(existing_memes)
    try:
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert music producer meme creator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()

        # NOTE: Show raw model output for debugging and parsing issues
        print("\n🔎 Raw OpenAI output:\n", content)

        lines = [line.strip() for line in content.split('\n') if line.strip()]
        csv_lines = []
        for line in lines:
            # NOTE: Accept only lines with exactly 9 commas (10 fields) to ensure correct CSV structure
            if line.count(',') == 9:
                csv_lines.append(line)
        return csv_lines[:5]
    except Exception as e:
        print(f"Error generating memes: {e}")
        return []


def parse_csv_line(line: str) -> Optional[Dict[str, str]]:
    """
    Parses a single CSV line into a meme dictionary.

    NOTE: Accepts only lines with exactly 10 fields to prevent errors from malformed model output.
    """
    try:
        reader = csv.reader([line])
        row = next(reader)
        if len(row) == 10:
            return {
                'header': row[0],
                'text1': row[1], 'text2': row[2], 'text3': row[3], 'text4': row[4],
                'joke': row[5],
                'img1': row[6], 'img2': row[7], 'img3': row[8], 'img4': row[9]
            }
    except Exception as e:
        print(f"Parse error: {e}")
    return None


def display_candidates(candidates: List[str]) -> List[Dict[str, str]]:
    parsed_candidates = []
    print("\n" + "="*80)
    print("🎵 NEW MEME CANDIDATES 🎵")
    print("="*80)
    for i, line in enumerate(candidates, 1):
        meme = parse_csv_line(line)
        if meme:
            parsed_candidates.append(meme)
            print(f"\n{i}. {meme['header']}")
            print(f"   [{meme['text1']}] [{meme['text2']}]")
            print(f"   [{meme['text3']}] [{meme['text4']}]")
            print(f"   💡 {meme['joke']}")
        else:
            print(f"\n{i}. [PARSING ERROR] {line}")
    print("\n" + "="*80)
    return parsed_candidates


def get_user_selection(num_candidates: int) -> tuple:
    while True:
        print(f"\nOptions:")
        print(f"• Type numbers to select (e.g., '1', '2', '1,2,3')")
        print(f"• Type 'try' to generate 5 new candidates")
        print(f"• Type 'quit' to exit")
        user_input = input("\nYour choice: ").strip().lower()
        if user_input == 'try':
            return ('try', [])
        elif user_input == 'quit':
            return ('quit', [])
        else:
            try:
                selections = []
                for part in user_input.split(','):
                    num = int(part.strip())
                    if 1 <= num <= num_candidates:
                        selections.append(num - 1)
                    else:
                        raise ValueError(f"Invalid number: {num}")
                if selections:
                    return ('select', selections)
                else:
                    print("❌ No valid selections found.")
            except ValueError as e:
                print(f"❌ Invalid input: {e}. Please try again.")


def save_selected_memes(selected_memes: List[Dict[str, str]]):
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # NOTE: Write header if file is newly created
        if not file_exists:
            writer.writerow(['header', 'text1', 'text2', 'text3', 'text4', 'joke', 'img1', 'img2', 'img3', 'img4'])
        for meme in selected_memes:
            writer.writerow([
                meme['header'], meme['text1'], meme['text2'], meme['text3'], meme['text4'],
                meme['joke'], meme['img1'], meme['img2'], meme['img3'], meme['img4']
            ])
    print(f"✅ Added {len(selected_memes)} new meme(s) to {CSV_FILE}")


def main():
    print("🎵 Music Producer Meme Generator 🎵")
    if not os.path.exists(CSV_FILE):
        print(f"❌ {CSV_FILE} not found!")
        return
    existing_memes = read_existing_memes()
    print(f"📚 Learned from {len(existing_memes)} existing memes")
    while True:
        print("\n🤖 Generating 5 new meme candidates...")
        candidates = generate_meme_candidates(existing_memes)
        if not candidates:
            print("❌ Failed to generate candidates. Try again? (y/n)")
            if input().lower() != 'y':
                break
            continue
        parsed_candidates = display_candidates(candidates)
        if not parsed_candidates:
            print("❌ No valid candidates generated. Trying again...")
            continue
        action, selections = get_user_selection(len(parsed_candidates))
        if action == 'quit':
            print("👋 Goodbye!")
            break
        elif action == 'try':
            continue
        elif action == 'select':
            selected_memes = [parsed_candidates[i] for i in selections]
            save_selected_memes(selected_memes)
            print(f"\n🎉 Successfully added your selected meme(s)!")
            print("Want to generate more? (y/n)")
            if input().strip().lower() != 'y':
                break


if __name__ == "__main__":
    main()
