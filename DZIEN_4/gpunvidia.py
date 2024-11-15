from numba import cuda
import numpy as np
from timeit import default_timer as timer

# Funkcja do uruchamiania na GPU, używająca CUDA
@cuda.jit
def gpufunction(a):
    idx = cuda.grid(1)
    if idx < a.size:
        a[idx] += 1

if __name__ == '__main__':
    # Rozmiar danych
    n = 100_000_000
    a = np.ones(n, dtype=np.float64)

    # Uruchomienie obliczeń na CPU
    start = timer()
    for i in range(n):
        a[i] += 1
    print(f'czas wykonania na CPU: {timer() - start}s')

    # Wybór drugiej karty graficznej (urządzenia)
    cuda.select_device(0)  # '1' oznacza drugą kartę (indeksowanie zaczyna się od 0)

    # Alokowanie danych na GPU
    d_a = cuda.to_device(a)

    # Uruchomienie obliczeń na GPU
    threads_per_block = 256
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

    start = timer()
    gpufunction[blocks_per_grid, threads_per_block](d_a)
    cuda.synchronize()  # Synchronizacja przed zmierzeniem czasu
    print(f'czas wykonania na GPU: {timer() - start}s')

    # Kopiowanie wyników z powrotem do pamięci CPU
    a = d_a.copy_to_host()
