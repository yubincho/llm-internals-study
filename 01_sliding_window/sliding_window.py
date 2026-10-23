"""
슬라이딩 윈도우 (Sliding Window)

창문을 옆으로 밀면서, 이미 계산한 걸 버리지 않고 다시 쓰는 방법.
"""


def max_sum_fixed(nums, k):
    """
    [고정 창문] 붙어 있는 k개의 합 중 가장 큰 값을 구함.

    예) max_sum_fixed([1, 3, 2, 5, 1, 1, 2], 3) → 10  ([3, 2, 5])

    시간: O(n)  (처음부터 다시 더하면 O(n * k))
    """
    if k <= 0 or k > len(nums):
        raise ValueError("k는 1 이상, 리스트 길이 이하여야 해요.")

    window_sum = sum(nums[:k])   # 첫 번째 창문의 합
    best = window_sum

    for i in range(k, len(nums)):
        window_sum += nums[i]        # 들어오는 숫자 더하기
        window_sum -= nums[i - k]    # 나가는 숫자 빼기
        best = max(best, window_sum)

    return best


def min_length_at_least(nums, target):
    """
    [애벌레 창문 - 짧은 구간] 합이 target 이상인 가장 짧은 구간의 길이를 구함.
    답이 없으면 0을 돌려줘요.

    예) min_length_at_least([2, 3, 1, 2, 4, 3], 7) → 2  ([4, 3])

    조건: 모든 숫자가 0 이상이어야 해요.
    시간: O(n)  (머리와 꼬리가 각각 한 번씩만 지나가)
    """
    left = 0
    window_sum = 0
    best = float("inf")

    for right in range(len(nums)):
        window_sum += nums[right]          # 머리 뻗기

        while window_sum >= target:        # 충분하면
            best = min(best, right - left + 1)   # 기록하고
            window_sum -= nums[left]       # 꼬리 당기기
            left += 1

    return 0 if best == float("inf") else best


def longest_unique_substring(s):
    """
    [애벌레 창문 - 긴 구간] 같은 글자가 겹치지 않는 가장 긴 조각의 길이를 구함.

    예) longest_unique_substring("pwwkew") → 3  ("wke")

    시간: O(n)
    """
    left = 0
    bag = set()   # 창문 안에 있는 글자 주머니
    best = 0

    for right in range(len(s)):
        while s[right] in bag:     # 겹치면
            bag.remove(s[left])    # 꼬리 당기기
            left += 1

        bag.add(s[right])          # 머리 뻗기
        best = max(best, right - left + 1)

    return best
