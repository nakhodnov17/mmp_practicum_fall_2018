def find_word_in_circle(circle, word):
    if not len(circle):
        return -1
    factor = len(word) // len(circle) + 2
    pos = (circle * factor).find(word)
    if pos >= 0:
        return pos, 1
    pos = (circle[::-1] * factor).find(word)
    if pos >= 0:
        return len(circle) - 1 - pos, -1
    return -1
