import math
import numpy as np


def safe_divide(numerator, denominator):
    try:
        index = numerator / denominator
    except ZeroDivisionError:
        return 0


def division(list1, list2):
    try:
        return len(list1) / len(list2)
    except ZeroDivisionError:
        return 0


def corrected_division(list1, list2):
    try:
        return len(list1) / math.sqrt(2 * len(list2))
    except ZeroDivisionError:
        return 0


def root_division(list1, list2):
    try:
        return len(list1) / math.sqrt(len(list2))
    except ZeroDivisionError:
        return 0


def squared_division(list1, list2):
    try:
        return len(list1) ** 2 / len(list2)
    except ZeroDivisionError:
        return 0


def log_division(list1, list2):
    try:
        return math.log(len(list1)) / math.log(len(list2))
    except ZeroDivisionError:
        return 0


def uber(list1, list2):
    try:
        return math.log(len(list1)) ** 2 / math.log(len(set(list2)) / len(list1))
    except ZeroDivisionError:
        return 0


def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1] + 1,
                    matrix[x, y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])