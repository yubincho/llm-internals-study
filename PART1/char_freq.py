"""표준국어대사전 표제어에서 한글 음절·자모 빈도를 센다."""

import csv
import sys
from collections import Counter
from pathlib import Path

DATA_PATH = Path(__file__).parent / "output" / "words.txt"
OUTPUT_DIR = Path(__file__).parent / "output"
RESULT_PATH = OUTPUT_DIR / "step01_result.csv"
TOP_N = 30

HANGUL_START = 0xAC00  # '가'
HANGUL_END = 0xD7A3  # '힣'
JUNG_COUNT = 21
JONG_COUNT = 28

CHOSEONG = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNGSEONG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONGSEONG = ["", *"ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"]


def load_words(path: Path) -> list[str]:
    # make_wordlist.py가 중복·어미·접사·옛한글을 이미 걸러 둔 목록을 쓴다.
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def is_hangul_syllable(ch: str) -> bool:
    return HANGUL_START <= ord(ch) <= HANGUL_END


def decompose(syllable: str) -> tuple[str, str, str]:
    index = ord(syllable) - HANGUL_START
    cho, rest = divmod(index, JUNG_COUNT * JONG_COUNT)
    jung, jong = divmod(rest, JONG_COUNT)
    return CHOSEONG[cho], JUNGSEONG[jung], JONGSEONG[jong]


def count_syllables(words: list[str]) -> Counter:
    return Counter(ch for word in words for ch in word if is_hangul_syllable(ch))


def count_jamo(syllables: Counter) -> dict[str, Counter]:
    counts = {"초성": Counter(), "중성": Counter(), "종성": Counter()}
    for syllable, n in syllables.items():
        cho, jung, jong = decompose(syllable)
        counts["초성"][cho] += n
        counts["중성"][jung] += n
        # 받침이 없는 음절이 많아서 '없음'도 하나의 값으로 보여 준다.
        counts["종성"][jong or "(없음)"] += n
    return counts


def print_ranking(title: str, counter: Counter, top_n: int) -> None:
    total = sum(counter.values())
    print(f"\n[{title}] 종류 {len(counter):,}개 / 총 {total:,}회")
    for rank, (item, n) in enumerate(counter.most_common(top_n), start=1):
        print(f"{rank:>3}. {item:<5} {n:>9,}  ({n / total:6.2%})")


def save_csv(path: Path, results: dict[str, Counter]) -> None:
    # 엑셀에서 한글이 깨지지 않도록 BOM이 붙는 utf-8-sig로 저장한다.
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["구분", "순위", "글자", "빈도", "비율"])
        for category, counter in results.items():
            total = sum(counter.values())
            for rank, (item, n) in enumerate(counter.most_common(), start=1):
                writer.writerow([category, rank, item, n, f"{n / total:.6f}"])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔에서 한글이 깨지지 않도록

    words = load_words(DATA_PATH)
    print(f"단어: {len(words):,}개")

    syllables = count_syllables(words)
    jamo = count_jamo(syllables)

    print_ranking("음절", syllables, TOP_N)
    for name, counter in jamo.items():
        print_ranking(name, counter, TOP_N)

    OUTPUT_DIR.mkdir(exist_ok=True)
    save_csv(RESULT_PATH, {"음절": syllables, **jamo})
    print(f"\n전체 결과 저장: {RESULT_PATH}")


if __name__ == "__main__":
    main()
