"""
소프트맥스 (Softmax)

점수를 '합이 1(100%)인 확률'로 바꾼다.
최고점 빼기 → 부풀리기(e^x) → 더하기 → 나누기
"""
import math
import random


def softmax(scores, temperature=1.0):
    """
    점수를 확률로 바꿔.

    예) softmax([2.0, 1.0, 0.1]) → [0.659, 0.242, 0.099]

    temperature: 낮으면 1등이 더 튀고(얌전), 높으면 다들 비슷해져(엉뚱).
    """
    if not scores:
        raise ValueError("점수가 하나도 없어요.")
    if temperature <= 0:
        raise ValueError("온도는 0보다 커야 해요. 1등만 고르려면 greedy를 쓰세요.")

    scaled = [s / temperature for s in scores]   # 0. 온도로 나누기
    biggest = max(scaled)                        # 1. 최고점 찾기
    if biggest == float("-inf"):
        raise ValueError("모든 점수가 -inf예요. 고를 수 있는 후보가 없어요.")

    exps = [math.exp(s - biggest) for s in scaled]   # 2. 최고점 빼고 부풀리기
    total = sum(exps)                                # 3. 더하기
    return [e / total for e in exps]                 # 4. 나누기


def log_softmax(scores):
    """
    softmax 결과에 log를 씌운 값을 '안전하게' 구해.

    log(softmax(x))를 그대로 하면, 아주 작은 확률이 0으로 반올림돼서
    log(0) 에러가 날 수 있어. 그래서 처음부터 log 상태로 계산해.
    """
    biggest = max(scores)
    log_total = biggest + math.log(sum(math.exp(s - biggest) for s in scores))
    return [s - log_total for s in scores]


def cross_entropy(scores, target_index):
    """
    틀린 정도(손실)를 재. 정답에 준 확률이 높을수록 0에 가까워져.

    예) cross_entropy([2.0, 1.0, 0.1], 0) → 0.417  (정답 확률 65.9%)
    """
    return -log_softmax(scores)[target_index]


def greedy(scores):
    """무조건 1등의 번호를 골라."""
    return max(range(len(scores)), key=lambda i: scores[i])


def top_k_filter(scores, k):
    """
    상위 k개만 남기고 나머지는 -inf로 바꿔.
    소프트맥스에 넣으면 -inf는 확률 0%가 돼.
    """
    if not 1 <= k <= len(scores):
        raise ValueError("k는 1 이상, 점수 개수 이하여야 해요.")

    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    keep = set(top)
    return [s if i in keep else float("-inf") for i, s in enumerate(scores)]


def sample(probs, rng=random):
    """확률대로 제비뽑기를 해서 번호 하나를 골라."""
    return rng.choices(range(len(probs)), weights=probs)[0]


if __name__ == "__main__":
    words = ["좋다", "맑다", "피자"]
    scores = [2.0, 1.0, 0.1]

    print("[온도별 확률]")
    for t in [0.5, 1.0, 2.0]:
        probs = softmax(scores, t)
        print(f"  T={t}: " + ", ".join(f"{w} {p:.1%}" for w, p in zip(words, probs)))

    print("\n[큰 숫자도 안 터져요]")
    print("  ", [round(p, 3) for p in softmax([1000, 999, 998])])

    print("\n[Top-2만 남기기]")
    probs = softmax(top_k_filter(scores, 2))
    print("  " + ", ".join(f"{w} {p:.1%}" for w, p in zip(words, probs)))

    print("\n[틀린 정도]")
    print(f"  정답이 '좋다'일 때: {cross_entropy(scores, 0):.3f}")
    print(f"  정답이 '피자'일 때: {cross_entropy(scores, 2):.3f}")

    print("\n[제비뽑기 5번]")
    rng = random.Random(42)   # 씨앗(seed)을 정하면 매번 같은 결과가 나와요
    probs = softmax(scores)
    print("  ", [words[sample(probs, rng)] for _ in range(5)])
