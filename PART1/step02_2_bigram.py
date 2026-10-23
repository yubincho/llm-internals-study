"""step02-2: 바로 앞 글자 하나를 보고 다음 글자를 뽑아 단어를 만든다 (바이그램).

02-1과 달리 단어 길이를 따로 정하지 않는다. 끝 표시($)가 뽑히면 단어가 끝나므로,
언제 끝낼지도 데이터에서 확률로 배운다.
"""

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from step02_1_unigram import OUTPUT_DIR, SAMPLE_COUNT, SEED, WORDS_PATH, load_words

TABLE_PATH = OUTPUT_DIR / "step02_2_bigram.csv"
RESULT_PATH = OUTPUT_DIR / "step02_2_result.txt"
NORANDOM_RESULT_PATH = OUTPUT_DIR / "step02_2_result_norandom.txt"

START = "^"
END = "$"
MAX_LENGTH = 10  # 1등만 고르면 같은 글자 고리를 끝없이 돌 수 있어 강제로 멈춘다


def build_bigram_table(words: list[str]) -> dict[str, Counter]:
    table: dict[str, Counter] = defaultdict(Counter)
    for word in words:
        chars = [START, *word, END]
        for prev, nxt in zip(chars, chars[1:]):
            table[prev][nxt] += 1
    return table


def save_table(path: Path, table: dict[str, Counter]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["앞 글자", "다음 글자", "횟수", "확률"])
        for prev in sorted(table):
            counter = table[prev]
            total = sum(counter.values())
            for nxt, n in counter.most_common():
                writer.writerow([prev, nxt, n, f"{n / total:.6f}"])


def pick_random(counter: Counter, rng: random.Random) -> str:
    return rng.choices(list(counter), weights=list(counter.values()))[0]


def pick_top(counter: Counter) -> str:
    return counter.most_common(1)[0][0]


def generate_word(table: dict[str, Counter], pick) -> tuple[str, bool]:
    """단어와, 끝 표시로 자연스럽게 끝났는지(False면 최대 길이에서 잘림)를 돌려준다."""
    letters: list[str] = []
    current = START
    while len(letters) < MAX_LENGTH:
        current = pick(table[current])
        if current == END:
            return "".join(letters), True
        letters.append(current)
    return "".join(letters), False


def format_line(word: str, finished: bool, dictionary: set[str]) -> str:
    marks = []
    if word in dictionary:
        marks.append("(사전에 있음)")
    if not finished:
        marks.append(f"(최대 {MAX_LENGTH}글자에서 잘림)")
    return f"{word}\t{' '.join(marks)}".rstrip()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔에서 한글이 깨지지 않도록
    no_random = "--norandom" in sys.argv[1:]

    words = load_words(WORDS_PATH)
    dictionary = set(words)
    table = build_bigram_table(words)
    save_table(TABLE_PATH, table)

    if no_random:
        pick = pick_top
        result_path = NORANDOM_RESULT_PATH
    else:
        rng = random.Random(SEED)
        pick = lambda counter: pick_random(counter, rng)  # noqa: E731
        result_path = RESULT_PATH

    generated = [generate_word(table, pick) for _ in range(SAMPLE_COUNT)]
    lines = [format_line(w, finished, dictionary) for w, finished in generated]
    hit = sum(w in dictionary for w, _ in generated)
    summary = f"사전에 있는 단어: {hit}/{SAMPLE_COUNT}"

    print("\n".join(lines))
    print(f"\n{summary}")
    result_path.write_text("\n".join(lines) + f"\n\n{summary}\n", encoding="utf-8")
    print(f"저장: {result_path}")
    print(f"전이 표: {TABLE_PATH}")


if __name__ == "__main__":
    main()
