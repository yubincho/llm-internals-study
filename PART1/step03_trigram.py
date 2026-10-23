"""step03: 앞 두 글자를 보고 다음 글자를 뽑아 단어를 만든다 (트라이그램, 세 글자 묶음).

02-2(앞 한 글자)와 같은 방식이고, 문맥만 한 글자 늘렸다.
시작은 '^^'로 채워서 첫 글자도 "앞 두 글자"를 보고 뽑게 한다.
"""

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from step02_1_unigram import OUTPUT_DIR, SAMPLE_COUNT, SEED, WORDS_PATH, load_words
from step02_2_bigram import END, MAX_LENGTH, START, format_line, pick_random, pick_top

TABLE_PATH = OUTPUT_DIR / "step03_trigram.csv"
RESULT_PATH = OUTPUT_DIR / "step03_result.txt"
NORANDOM_RESULT_PATH = OUTPUT_DIR / "step03_result_norandom.txt"

CONTEXT_SIZE = 2  # 앞에서 볼 글자 수
MIN_MEANINGFUL_LENGTH = 2  # 한 글자 단어는 거의 다 사전에 있어 적중률을 부풀린다


def build_trigram_table(words: list[str]) -> dict[str, Counter]:
    table: dict[str, Counter] = defaultdict(Counter)
    for word in words:
        chars = [START] * CONTEXT_SIZE + [*word, END]
        for i in range(CONTEXT_SIZE, len(chars)):
            context = "".join(chars[i - CONTEXT_SIZE : i])
            table[context][chars[i]] += 1
    return table


def save_table(path: Path, table: dict[str, Counter]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["앞 두 글자", "다음 글자", "횟수", "확률"])
        for context in sorted(table):
            counter = table[context]
            total = sum(counter.values())
            for nxt, n in counter.most_common():
                writer.writerow([context, nxt, n, f"{n / total:.6f}"])


def generate_word(table: dict[str, Counter], pick) -> tuple[str, bool]:
    """단어와, 끝 표시로 자연스럽게 끝났는지(False면 최대 길이에서 잘림)를 돌려준다."""
    letters: list[str] = []
    context = START * CONTEXT_SIZE
    while len(letters) < MAX_LENGTH:
        nxt = pick(table[context])
        if nxt == END:
            return "".join(letters), True
        letters.append(nxt)
        context = context[1:] + nxt
    return "".join(letters), False


def summarize(generated: list[tuple[str, bool]], dictionary: set[str]) -> str:
    hit = sum(w in dictionary for w, _ in generated)
    long_words = [w for w, _ in generated if len(w) >= MIN_MEANINGFUL_LENGTH]
    long_hit = sum(w in dictionary for w in long_words)
    return (
        f"사전에 있는 단어: {hit}/{len(generated)}\n"
        f"{MIN_MEANINGFUL_LENGTH}글자 이상만: {long_hit}/{len(long_words)}"
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔에서 한글이 깨지지 않도록
    no_random = "--norandom" in sys.argv[1:]

    words = load_words(WORDS_PATH)
    dictionary = set(words)
    table = build_trigram_table(words)
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
    summary = summarize(generated, dictionary)

    print("\n".join(lines))
    print(f"\n{summary}")
    result_path.write_text("\n".join(lines) + f"\n\n{summary}\n", encoding="utf-8")
    print(f"저장: {result_path}")
    print(f"전이 표: {TABLE_PATH}")


if __name__ == "__main__":
    main()
