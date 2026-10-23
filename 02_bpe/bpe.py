"""
BPE (Byte Pair Encoding)

가장 자주 붙어 다니는 두 조각을 하나로 합치고, 이걸 반복한다.
새 단어는 합친 순서 그대로 따라 해서 자른다.
"""
from collections import Counter


def count_pairs(words):
    """
    모든 짝꿍(나란히 붙은 두 조각)이 몇 번 나왔는지 세.

    words: {('h', 'u', 'g'): 3, ...}  단어(조각 묶음) → 나온 횟수
    """
    pairs = Counter()
    for word, count in words.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += count   # 단어 개수만큼 더하기
    return pairs


def merge_word(word, pair):
    """
    단어 하나 안에서 짝꿍(pair)을 찾아 하나로 합쳐.

    예) merge_word(('h', 'u', 'g'), ('u', 'g')) → ('h', 'ug')
    """
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            new_word.append(word[i] + word[i + 1])
            i += 2   # 합쳤으니까 두 칸 건너뛰기
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def train(text, num_merges, verbose=False):
    """
    BPE 공부시키기: 합치기 규칙(merges)을 순서대로 만들어.

    더 합칠 짝꿍이 없으면 num_merges보다 일찍 멈춰.
    동점이면 먼저 나온 짝꿍을 골라. (Python dict는 넣은 순서를 지킴)
    """
    words = {tuple(w): c for w, c in Counter(text.split()).items()}
    merges = []   # 합친 순서를 적는 공책

    for _ in range(num_merges):
        pairs = count_pairs(words)
        if not pairs:   # 모든 단어가 한 덩어리가 됐으면 멈추기
            break

        best = max(pairs, key=pairs.get)   # 1등 짝꿍
        merges.append(best)
        words = {merge_word(w, best): c for w, c in words.items()}

        if verbose:
            print(f"{best[0]} + {best[1]} → {best[0] + best[1]}  ({pairs[best]}번)")

    return merges


def encode(word, merges):
    """
    새 단어를 블록으로 잘라요. 공책에 적힌 순서대로 합치기!

    예) encode("unhug", merges) → ['un', 'hug']
    """
    pieces = tuple(word)
    for pair in merges:
        pieces = merge_word(pieces, pair)
    return list(pieces)


def decode(pieces):
    """블록들을 다시 붙여서 원래 단어로 되돌려."""
    return "".join(pieces)


def build_vocab(text, merges):
    """
    블록마다 번호를 붙여. (토큰 사전)

    1칸 블록은 가나다순으로 먼저, 합친 블록은 합친 순서대로 뒤에 붙여.
    """
    base = sorted(set(text.replace(" ", "")))
    merged = [a + b for a, b in merges]
    return {token: i for i, token in enumerate(base + merged)}


def encode_to_ids(word, merges, vocab):
    """단어를 블록으로 자른 다음 번호로 바꿔. LLM이 실제로 먹는 건 이 번호."""
    ids = []
    for piece in encode(word, merges):
        if piece not in vocab:
            raise ValueError(f"사전에 없는 조각이에요: {piece!r}")
        ids.append(vocab[piece])
    return ids


if __name__ == "__main__":
    text = "hug hug hug pug pug pun pun bun hugs"
    merges = train(text, 3, verbose=True)
    vocab = build_vocab(text, merges)

    for w in ["bug", "unhug"]:
        pieces = encode(w, merges)
        print(f"{w} → {pieces} → {encode_to_ids(w, merges, vocab)}")
