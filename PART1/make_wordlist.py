"""표준국어대사전 CSV(낱말,품사)에서 낱말만 뽑아 한 줄에 하나씩 저장한다."""

import csv
import sys
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "kr_korean.csv"
OUTPUT_PATH = Path(__file__).parent / "output" / "words.txt"

MORPHEME_BOUNDARY = "-"  # 사전의 형태소 경계 표시 (냉수욕-하다)
FULLWIDTH_BOUNDARY = "－"  # 일부 표제어는 전각 하이픈으로 들어 있다 (카뜽－하다)
SPACING_MARK = "^"  # 사전의 띄어쓰기 허용 표시 (냉수성^어류)
PRIVATE_USE_START, PRIVATE_USE_END = 0xE000, 0xF8FF


def is_affix(word: str) -> bool:
    # '-가', '가-' 같은 어미·접사는 혼자 쓰이는 단어가 아니다.
    return word.startswith(MORPHEME_BOUNDARY) or word.endswith(MORPHEME_BOUNDARY)


def has_private_use_char(word: str) -> bool:
    # 옛한글 등이 사전 고유의 사용자 정의 영역 코드로 들어 있어 다른 환경에선 깨진다.
    return any(PRIVATE_USE_START <= ord(ch) <= PRIVATE_USE_END for ch in word)


def normalize(word: str) -> str:
    for boundary in (MORPHEME_BOUNDARY, FULLWIDTH_BOUNDARY):
        word = word.replace(boundary, "")
    return word.replace(SPACING_MARK, " ").strip()


def load_words(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        raw = (row[0] for row in csv.reader(f) if row)
        words = {normalize(w) for w in raw if not is_affix(w) and not has_private_use_char(w)}
    return sorted(w for w in words if w)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔에서 한글이 깨지지 않도록

    words = load_words(DATA_PATH)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(words) + "\n", encoding="utf-8")
    print(f"{len(words):,}개 저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
