import line_profiler

from src.spatial_index.common_utils import quick_sort


def fun(a, b, c):
    i = 0
    j = 0
    k = 0
    while i < 8 and j < 6:
        if a[i] > b[j]:
            c[k] = b[j]
            j += 1
        else:
            c[k] = a[i]
            i += 1
        k += 1
    while i < 8:
        c[k] = a[i]
        k += 1
        i += 1
    while j < 6:
        c[k] = b[j]
        k += 1
        j += 1


def fun2(a, b):
    i = 0
    j = 0
    while j < 6:
        if a[i] >= b[j]:
            a.insert(i, b[j])
            j += 1
        i += 1
    while j < 6:
        a.append(b[j])
    del a[14:20]


if __name__ == '__main__':
    def fun100():
        for i in range(1000):
            a = [[5, 5, 5], [1, 1, 1], [2, 2, 2], [5, 5, 5], [4, 4, 4], [3, 3, 3], [0, 0, 0], [5, 5, 5]]
            quick_sort(a, 2, 0, 7)
            b = [[5, 5, 5], [1, 1, 1], [5, 5, 5], [4, 4, 4], [3, 3, 3], [0, 0, 0]]
            quick_sort(b, 2, 0, 5)
            a.extend(b)
            quick_sort(a, 2, 0, 13)
            print(a)
            a = [[5, 5, 5], [1, 1, 1], [2, 2, 2], [5, 5, 5], [4, 4, 4], [3, 3, 3], [0, 0, 0], [5, 5, 5]]
            quick_sort(a, 2, 0, 7)
            c = [None] * 14
            b = [[5, 5, 5], [1, 1, 1], [5, 5, 5], [4, 4, 4], [3, 3, 3], [0, 0, 0]]
            quick_sort(b, 2, 0, 5)
            fun(a, b, c)
            print(c)
            a = [[5, 5, 5], [1, 1, 1], [2, 2, 2], [5, 5, 5], [4, 4, 4], [3, 3, 3], [0, 0, 0], [5, 5, 5]]
            quick_sort(a, 2, 0, 7)
            a.extend([None] * 6)
            b = [[5, 5, 5], [1, 1, 1], [5, 5, 5], [4, 4, 4], [3, 3, 3], [0, 0, 0]]
            quick_sort(b, 2, 0, 5)
            fun2(a, b)
            print(c)
profile = line_profiler.LineProfiler(fun100)
profile.enable()
fun100()
profile.disable()
profile.print_stats()
