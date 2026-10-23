from bpe import (
    count_pairs,
    merge_word,
    train,
    encode,
    decode,
    build_vocab,
    encode_to_ids,
)

TEXT = "hug hug hug pug pug pun pun bun hugs"


def test_count_pairs():
    pairs = count_pairs({("h", "u", "g"): 3, ("p", "u", "g"): 2})
    assert pairs[("u", "g")] == 5     # 3 + 2
    assert pairs[("h", "u")] == 3
    assert pairs[("p", "u")] == 2


def test_merge_word():
    assert merge_word(("h", "u", "g"), ("u", "g")) == ("h", "ug")
    assert merge_word(("a", "b", "a", "b"), ("a", "b")) == ("ab", "ab")  # 여러 번
    assert merge_word(("x", "y"), ("u", "g")) == ("x", "y")              # 없으면 그대로


def test_train_order():
    merges = train(TEXT, 3)
    assert merges == [("u", "g"), ("h", "ug"), ("u", "n")]


def test_train_stops_early():
    # "aaa"는 2번 합치면 한 덩어리가 돼서 더 합칠 게 없어요
    merges = train("aaa", 10)
    assert len(merges) == 2
    assert encode("aaa", merges) == ["aaa"]


def test_encode_unseen_words():
    merges = train(TEXT, 3)
    assert encode("bug", merges) == ["b", "ug"]
    assert encode("unhug", merges) == ["un", "hug"]
    assert encode("", merges) == []   # 빈 글자


def test_decode_round_trip():
    # 잘랐다가 다시 붙이면 원래 단어로 돌아와야 해요
    merges = train(TEXT, 3)
    for w in ["hug", "bug", "unhug", "pugs", "b"]:
        assert decode(encode(w, merges)) == w


def test_vocab_and_ids():
    merges = train(TEXT, 3)
    vocab = build_vocab(TEXT, merges)
    assert len(vocab) == 10            # 1칸 블록 7개 + 새 블록 3개
    assert encode_to_ids("unhug", merges, vocab) == [vocab["un"], vocab["hug"]]


def test_unknown_character():
    merges = train(TEXT, 3)
    vocab = build_vocab(TEXT, merges)
    try:
        encode_to_ids("cat", merges, vocab)   # c, a, t는 사전에 없어요
        assert False, "에러가 나야 해요!"
    except ValueError:
        pass


if __name__ == "__main__":
    test_count_pairs()
    test_merge_word()
    test_train_order()
    test_train_stops_early()
    test_encode_unseen_words()
    test_decode_round_trip()
    test_vocab_and_ids()
    test_unknown_character()
    print("모든 테스트 통과! ✅")
