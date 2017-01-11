import random

def random_offset(amp):
    return random.uniform(-amp, amp)

def levenshtein_distance(a, b):
    cost = 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

  # test if last characters of the strings match
    if a[-1] == b[-1]:
        cost = 0
    else:
        cost = 1

    return min(levenshtein_distance(a[:-1], b) + 1, levenshtein_distance(a, b[:-1]) + 1, levenshtein_distance(a[:-1], b[:-1]) + cost)

if __name__ == "__main__":
    assert levenshtein_distance("123", "124") == 1
    assert levenshtein_distance("13", "123") == 1
    assert levenshtein_distance("123456", "1245") == 2
    assert levenshtein_distance("123456", "34") == 4
