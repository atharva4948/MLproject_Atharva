import time
import random

def merge(arr, left, mid, right):
    n1 = mid - left + 1
    n2 = right - mid
    L = arr[left:mid + 1]
    R = arr[mid + 1:right + 1]
    i = j = 0
    k = left
    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1
    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1
    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1

def merge_sort(arr, left, right):
    if left < right:
        mid = (left + right) // 2
        merge_sort(arr, left, mid)
        merge_sort(arr, mid + 1, right)
        merge(arr, left, mid, right)

import threading
import time
import random

def merge(arr, left, mid, right):
    n1 = mid - left + 1
    n2 = right - mid
    L = arr[left:mid + 1]
    R = arr[mid + 1:right + 1]
    i = j = 0
    k = left
    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1
    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1
    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1

def threaded_merge_sort(arr, left, right):
    if left < right:
        mid = (left + right) // 2
        left_thread = threading.Thread(target=threaded_merge_sort, args=(arr, left, mid))
        right_thread = threading.Thread(target=threaded_merge_sort, args=(arr, mid + 1, right))
        left_thread.start()
        right_thread.start()
        left_thread.join()
        right_thread.join()
        merge(arr, left, mid, right)

sizes = [1000, 5000, 10000, 50000, 100000]

for size in sizes:
    arr = [random.randint(0, 1000000) for _ in range(size)]
    arr_copy = arr.copy()

    start = time.time()
    merge_sort(arr, 0, len(arr) - 1)
    end = time.time()
    print(f"Merge Sort (n={size}) : {end - start:.5f} sec")

    start = time.time()
    threaded_merge_sort(arr_copy, 0, len(arr_copy) - 1)
    end = time.time()
    print(f"Multithreaded Merge Sort (n={size}) : {end - start:.5f} sec\n")
