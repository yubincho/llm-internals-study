from sliding_window import (
    max_sum_fixed,
    min_length_at_least,
    longest_unique_substring,
)


def test_max_sum_fixed():
    assert max_sum_fixed([1, 3, 2, 5, 1, 1, 2], 3) == 10
    assert max_sum_fixed([5], 1) == 5                 # 상자 1개
    assert max_sum_fixed([1, 2, 3], 3) == 6           # 창문 = 전체
    assert max_sum_fixed([-1, -2, -3], 2) == -3       # 음수만 있어도 OK


def test_max_sum_fixed_wrong_k():
    for bad_k in [0, 4]:
        try:
            max_sum_fixed([1, 2, 3], bad_k)
            assert False, "에러가 나야 해요!"
        except ValueError:
            pass   # 에러가 났으면 통과


def test_min_length_at_least():
    assert min_length_at_least([2, 3, 1, 2, 4, 3], 7) == 2
    assert min_length_at_least([1, 4, 4], 4) == 1      # 상자 하나로 충분
    assert min_length_at_least([1, 1, 1], 10) == 0     # 답이 없어요
    assert min_length_at_least([], 5) == 0             # 빈 리스트


def test_longest_unique_substring():
    assert longest_unique_substring("pwwkew") == 3
    assert longest_unique_substring("abcabcbb") == 3
    assert longest_unique_substring("bbbbb") == 1
    assert longest_unique_substring("") == 0           # 빈 글자


if __name__ == "__main__":
    test_max_sum_fixed()
    test_max_sum_fixed_wrong_k()
    test_min_length_at_least()
    test_longest_unique_substring()
    print("모든 테스트 통과! ✅")
