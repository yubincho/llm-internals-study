"""step04: 유니그램·바이그램·트라이그램으로 단어를 많이 뽑아 비율로 비교한다.

20개로는 운에 따라 결과가 크게 흔들려서, 모델마다 같은 개수를 뽑아 지표를 숫자로 잰다.
"""

import csv
import random
import sys
from collections import Counter
from pathlib import Path

import step02_1_unigram as unigram
import step02_2_bigram as bigram
import step03_trigram as trigram
from step02_1_unigram import OUTPUT_DIR, SEED, WORDS_PATH, load_words

RESULT_PATH = OUTPUT_DIR / "step04_result.csv"
SAMPLE_COUNT = 1_000
MIN_MEANINGFUL_LENGTH = 2  # 한 글자 단어는 거의 다 사전에 있어 적중률을 부풀린다


def build_generators(words: list[str], rng: random.Random) -> dict:
    """모델 이름 → (단어, 자연스럽게 끝났는지)를 돌려주는 함수."""
    syllable_freq = unigram.load_syllable_freq(unigram.FREQ_PATH)
    length_freq = Counter(len(w) for w in words)
    bigram_table = bigram.build_bigram_table(words)
    trigram_table = trigram.build_trigram_table(words)
    pick = lambda counter: bigram.pick_random(counter, rng)  # noqa: E731

    return {
        # 유니그램은 길이를 미리 정하므로 잘리는 일이 없다.
        "유니그램 (문맥 없음)": lambda: (unigram.generate_word(syllable_freq, length_freq, rng), True),
        "바이그램 (앞 1글자)": lambda: bigram.generate_word(bigram_table, pick),
        "트라이그램 (앞 2글자)": lambda: trigram.generate_word(trigram_table, pick),
    }


def measure(samples: list[tuple[str, bool]], dictionary: set[str]) -> dict[str, float]:
    words = [w for w, _ in samples]
    total = len(words)
    long_words = [w for w in words if len(w) >= MIN_MEANINGFUL_LENGTH]
    return {
        "사전 적중률": sum(w in dictionary for w in words) / total,
        "사전 적중률(2글자 이상)": sum(w in dictionary for w in long_words) / len(long_words),
        "새 단어 비율": sum(w not in dictionary for w in words) / total,
        "서로 다른 단어 비율": len(set(words)) / total,
        "한 글자 단어 비율": sum(len(w) == 1 for w in words) / total,
        "평균 길이": sum(map(len, words)) / total,
        "잘림 비율": sum(not finished for _, finished in samples) / total,
    }


def measure_dictionary(words: list[str]) -> dict[str, float]:
    # 생성 결과가 실제 단어 모양과 얼마나 닮았는지 볼 기준선
    total = len(words)
    return {
        "한 글자 단어 비율": sum(len(w) == 1 for w in words) / total,
        "평균 길이": sum(map(len, words)) / total,
    }


def format_value(metric: str, value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}" if metric == "평균 길이" else f"{value:.1%}"


def print_table(results: dict[str, dict[str, float]], metrics: list[str]) -> None:
    names = list(results)
    print(f"{'지표':<20}" + "".join(f"{n:>16}" for n in names))
    for metric in metrics:
        cells = "".join(f"{format_value(metric, results[n].get(metric)):>16}" for n in names)
        print(f"{metric:<20}{cells}")


def save_csv(path: Path, results: dict[str, dict[str, float]], metrics: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["지표", *results])
        for metric in metrics:
            writer.writerow([metric, *(format_value(metric, r.get(metric)) for r in results.values())])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔에서 한글이 깨지지 않도록

    words = load_words(WORDS_PATH)
    dictionary = set(words)
    rng = random.Random(SEED)

    results: dict[str, dict[str, float]] = {}
    for name, generate in build_generators(words, rng).items():
        samples = [generate() for _ in range(SAMPLE_COUNT)]
        results[name] = measure(samples, dictionary)
    results["(참고) 실제 사전"] = measure_dictionary(words)

    metrics = list(next(iter(results.values())))
    print(f"모델마다 {SAMPLE_COUNT:,}개씩 생성\n")
    print_table(results, metrics)
    save_csv(RESULT_PATH, results, metrics)
    print(f"\n저장: {RESULT_PATH}")


if __name__ == "__main__":
    main()
