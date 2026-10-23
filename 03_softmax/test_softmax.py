import math
import random

from softmax import (
    softmax,
    log_softmax,
    cross_entropy,
    greedy,
    top_k_filter,
    sample,
)


def close(a, b, tol=1e-3):
    """두 숫자가 거의 같은지 확인해요. (소수는 아주 조금씩 오차가 있어서요)"""
    return abs(a - b) < tol


def test_basic_values():
    probs = softmax([2.0, 1.0, 0.1])
    assert all(close(p, e) for p, e in zip(probs, [0.659, 0.242, 0.099]))


def test_sum_is_one():
    for scores in [[1, 2, 3], [-5, 0, 5], [0.1], [7, 7, 7, 7]]:
        assert close(sum(softmax(scores)), 1.0, 1e-9)


def test_same_scores_same_probs():
    assert all(close(p, 0.25) for p in softmax([3, 3, 3, 3]))


def test_shift_does_not_change_result():
    # 모두에게 같은 숫자를 빼도 결과는 같아 (계단 같이 내려가기)
    a = softmax([1, 2, 3])
    b = softmax([101, 102, 103])
    assert all(close(x, y, 1e-9) for x, y in zip(a, b))


def test_big_numbers_do_not_explode():
    probs = softmax([1000, 999, 998])
    assert all(not math.isnan(p) for p in probs)
    assert close(probs[0], 0.665)


def test_temperature():
    low = softmax([2.0, 1.0, 0.1], 0.5)
    high = softmax([2.0, 1.0, 0.1], 2.0)
    assert low[0] > high[0]      # 온도가 낮으면 1등이 더 튀어
    assert low[2] < high[2]      # 온도가 높으면 꼴찌도 기회가 생겨


def test_wrong_inputs():
    for bad in [lambda: softmax([]),
                lambda: softmax([1, 2], 0),
                lambda: softmax([float("-inf")] * 3),
                lambda: top_k_filter([1, 2, 3], 0)]:
        try:
            bad()
            assert False, "에러가 나야 해요!"
        except ValueError:
            pass


def test_log_softmax_is_safe():
    # 그냥 log(softmax)를 하면 log(0) 에러가 나는 경우
    result = log_softmax([0, -1000])
    assert close(result[0], 0.0)
    assert close(result[1], -1000.0)


def test_cross_entropy():
    assert close(cross_entropy([2.0, 1.0, 0.1], 0), 0.417)
    assert close(cross_entropy([2.0, 1.0, 0.1], 2), 2.317)
    # 정답 점수가 압도적이면 틀린 정도는 거의 0
    assert cross_entropy([100, 0, 0], 0) < 1e-6


def test_greedy():
    assert greedy([0.1, 5.0, 2.0]) == 1


def test_top_k_filter():
    filtered = top_k_filter([2.0, 1.0, 0.1], 2)
    probs = softmax(filtered)
    assert probs[2] == 0.0                   # 3등은 0%
    assert close(sum(probs), 1.0, 1e-9)      # 남은 둘이 100%를 나눠 가져


def test_sample_follows_probs():
    rng = random.Random(0)
    probs = softmax([2.0, 1.0, 0.1])
    n = 10_000
    picks = [sample(probs, rng) for _ in range(n)]
    ratio = picks.count(0) / n
    assert 0.63 < ratio < 0.69               # 약 65.9%쯤 뽑혀야 해


def test_sample_never_picks_zero_prob():
    rng = random.Random(0)
    probs = softmax(top_k_filter([2.0, 1.0, 0.1], 1))   # 1등만 남기기
    assert all(sample(probs, rng) == 0 for _ in range(100))


if __name__ == "__main__":
    test_basic_values()
    test_sum_is_one()
    test_same_scores_same_probs()
    test_shift_does_not_change_result()
    test_big_numbers_do_not_explode()
    test_temperature()
    test_wrong_inputs()
    test_log_softmax_is_safe()
    test_cross_entropy()
    test_greedy()
    test_top_k_filter()
    test_sample_follows_probs()
    test_sample_never_picks_zero_prob()
    print("모든 테스트 통과! ✅")
