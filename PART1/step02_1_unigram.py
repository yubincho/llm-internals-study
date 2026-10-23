"""step02-1: step01의 음절 빈도만 보고 글자를 무작위로 뽑아 단어를 만든다.

앞 글자를 전혀 보지 않으므로(유니그램), 빈도만으로 단어가 만들어지는지 확인하는 기준선이다.
"""

import csv
import random
import sys
from collections import Counter
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
FREQ_PATH = OUTPUT_DIR / "step01_result.csv"
WORDS_PATH = OUTPUT_DIR / "words.txt"
RESULT_PATH = OUTPUT_DIR / "step02_1_result.txt"
NORANDOM_RESULT_PATH = OUTPUT_DIR / "step02_result_norandom.txt"

SAMPLE_COUNT = 20
SEED = 42  # 같은 결과를 다시 볼 수 있도록 고정


def load_syllable_freq(path: Path) -> Counter:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return Counter({row["글자"]: int(row["빈도"]) for row in csv.DictReader(f) if row["구분"] == "음절"})


def load_words(path: Path) -> list[str]:
    # 띄어쓰기가 든 표제어는 여러 단어라서 길이 분포를 왜곡한다.
    return [w for w in path.read_text(encoding="utf-8").splitlines() if w and " " not in w]


def generate_word(syllable_freq: Counter, length_freq: Counter, rng: random.Random) -> str:
    # 단어 길이도 실제 사전의 길이 분포에서 뽑아야 결과를 사전 단어와 비교할 수 있다.
    length = rng.choices(list(length_freq), weights=list(length_freq.values()))[0]
    letters = rng.choices(list(syllable_freq), weights=list(syllable_freq.values()), k=length)
    return "".join(letters)


def generate_word_greedy(syllable_freq: Counter, length_freq: Counter) -> str:
    # 무작위 없이 매번 1등만 고른다 (temperature를 0에 가깝게 낮춘 극단).
    length = length_freq.most_common(1)[0][0]
    top_syllable = syllable_freq.most_common(1)[0][0]
    return top_syllable * length


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔에서 한글이 깨지지 않도록
    no_random = "--norandom" in sys.argv[1:]

    syllable_freq = load_syllable_freq(FREQ_PATH)
    words = load_words(WORDS_PATH)
    dictionary = set(words)
    length_freq = Counter(len(w) for w in words)

    if no_random:
        generated = [generate_word_greedy(syllable_freq, length_freq) for _ in range(SAMPLE_COUNT)]
    else:
        rng = random.Random(SEED)
        generated = [generate_word(syllable_freq, length_freq, rng) for _ in range(SAMPLE_COUNT)]
    result_path = NORANDOM_RESULT_PATH if no_random else RESULT_PATH

    lines = [f"{w}\t{'(사전에 있음)' if w in dictionary else ''}".rstrip() for w in generated]
    hit = sum(w in dictionary for w in generated)
    summary = f"사전에 있는 단어: {hit}/{SAMPLE_COUNT}"

    print("\n".join(lines))
    print(f"\n{summary}")
    result_path.write_text("\n".join(lines) + f"\n\n{summary}\n", encoding="utf-8")
    print(f"저장: {result_path}")


if __name__ == "__main__":
    main()
