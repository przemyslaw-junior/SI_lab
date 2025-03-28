from collections import deque
import time
import pandas as pd
import matplotlib.pyplot as plt
import heapq


checked_state = 0

# sprawdzenie  ataków hetmanów.
def is_safe_solution(queens):
    for i in range(len(queens)):
        for j in range(i +1, len(queens)):
            qx1, qy1 = queens[i]
            qx2, qy2 = queens[j]
            if qx1 == qx2 or qy1 == qy2 or abs(qx1 - qx2) == abs(qy1 - qy2):
                return False
    return True

# zliczanie par hetmanów które się zwajemnie akatują (kolumna, przekątna)
def count_attacks(state):
    attacks = 0
    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            qx1, qy1 = state[i]
            qx2, qy2 = state[j]
            if qx1 == qx2 or qy1 == qy2 or abs(qx1 - qx2) == abs(qy1 - qy2):
               attacks += 1
    return attacks 


# metoda generujaca potomków
def generate_children(state, N):
    global checked_state
    # index wiersza , do którego dodajemy nowego hetmana -> poziom w drzewie
    x = len(state)

    # jeżeli mamy juz n-hetmanów, nie tworzymy kolejnych dzieci
    if x == N:
        return [state]
    children = []
    # unikanie powtórzeń - optymalizacja
    taken_columns = {qy for _, qy in state}

    for y in range(N):
        # dodanie hetmanów do kolumn, które nie sa zajęte
        if y not in taken_columns:
            # tworzenie nowego stanu
            new_state = state + [(x, y)]
            checked_state += 1
            # print(f"kolejne dziecko {checked_state}: {new_state}")
            children.append(new_state)
    return children


# HEURYSTYKI

# H1 -> wybór hetmanów z pierwszeństwem dla środkowych wierszy.
# h1(s) = (N - 1) * sum(waga wiersza)
def heuristic_H1(state, N):
    num_queens = len(state)
    sum_weights_row = 0

    for (x,_) in state:
        # poprawa indeksowania od 1
        row_i = x + 1
        # sprawdzenie gdzie jest obeecny wiersz.
        # w pierwszej połowie, waga = n - row_i + 1
        if row_i <= N // 2:
            sum_weights_row += (N - row_i + 1)
        # druga połowa, waga row_i
        else:
            sum_weights_row += row_i

    return (N - num_queens) * sum_weights_row

# H2 -> w pierwszej kolejności należy rozwijac węzły z najmniejszą liczbą atakujących sie hetmanów
# oraz z najwieksza liczbąjuż wstawionych hetmanów (zaawansowane rozwiązania)
# h2(s) = count_attacks(s) +(N + num_queens)
def heuristic_H2(state, N):
    num_queens = len(state)
    return count_attacks(state) + (N - num_queens)


# HDOD -> sumowanie różnic miedzy odległościami ( Manhattan) 3=odległość ruchu
def heuristic_HDOD(state, N):
    total_diff = 0
    for i in range(len(state)):
        for j in range(len(state)):
            qx1, qy1 = state[i]
            qx2, qy2 = state[j]
            dist = abs(qx1 - qx2) + abs (qy1 - qy2)
            total_diff += abs(dist - 3)
    return total_diff


# wybieranie heurystyki
def get_heuristic(state, N, heuristic):
    if heuristic == 'H1':
        return heuristic_H1(state, N)
    elif heuristic == 'H2':
        return heuristic_H2(state, N)
    elif heuristic.upper() == 'HDOD':
        return heuristic_HDOD(state, N)
    else:
        return 0
    


# algorytm Best First Search (BestFS)
def bestFS_n_hetman(N, heuristic='H1', debug=False):
    global checked_state
    checked_state = 0

    start_time = time.perf_counter()

    # kolejka priorytetowa
    Open = []
    Closed = set()

    # stan początkowy
    initial_state = []
    initial_H = get_heuristic(initial_state, N, heuristic)
    heapq.heappush(Open, (initial_H, initial_state))

    while Open:
        # pobranie stanu o najmniejszej wartości heurystyki
        current_h, state = heapq.heappop(Open)
        checked_state += 1

        if debug:
            print(f"Sprawdzam stan: {state}, h={current_h}")

        # sprawdzenie czy mamy już max liczbe hetmanów i jest bezpiecznie
        if len(state) == N and is_safe_solution(state):
            end_time = time.perf_counter() - start_time
            return state, end_time, checked_state, len(Open), current_h
        
        # dodanie stanu do Close, nie przetwarzamy stanu ponownie (optymalizaja)
        if tuple(state) in Closed:
            continue
        Closed.add(tuple(state))

        children = generate_children(state, N)
        for child in children:
            # jeżeli nie odwiedzony
            if tuple(child) not in Closed:
               heuristic_val = get_heuristic(child, N, heuristic)
               # dodanie do kolejki priorytetowej
               heapq.heappush(Open, (heuristic_val, child)) 

    # jeżeli nie ma rozwiazań zwracamy pusty wynik
    return [], 0, checked_state, len(Open), None


# algorytm BFS kolejka -> przeszukuje drzewo poziomami
def bfs_n_hetman(N):
    global checked_state
    checked_state = 0

    start_time = time.perf_counter()
    # kolejka FIFO BFS
    Open = deque([[]])
    Closed = set()

    while Open:
        # pobranie pierwszego stanu zkolejki
        state = Open.popleft()
        checked_state += 1

        # print(f"bfs sprawdzenie stanu {checked_state_bfs}: {state}")
        # jezeli mamy N hetmanów, znaleziono pełne rozwiązanie
        if len(state) == N and is_safe_solution(state):
            end_time = time.perf_counter() - start_time
            open_states = len(Open)
            return state, end_time, checked_state, open_states

        Closed.add(tuple(state))

        children = generate_children(state, N)

        # dodanie nowego stanu do kolejki
        for child in children:
            if tuple(child) not in Closed and child not in Open:
                # print(f"bfs dodanie dziecka do open: {child}")
                Open.append(child)
        # print(f"Open: {list(Open)}")
        # print(f"Closed: {Closed}\n")

    return [], 0, checked_state, len(Open)


# algorytm DFS stos -> przeszukuje drzewo w głąb
def dfs_n_hetman(N):
    global checked_state
    checked_state = 0

    start_time = time.perf_counter()
    # kolejka LIFO BFS
    Open = [[]]
    Closed = set()

    while Open:
        # pobranie pierwszego stanu z kolejki
        state = Open.pop()
        checked_state += 1
        # print(f"dsf sprawdzenie stanu: {checked_state_dfs}: {state}")
        # jezeli mamy N hetmanów, znaleziono pełne rozwiązanie
        if len(state) == N and  is_safe_solution(state):

            end_time = time.perf_counter() - start_time
            open_states = len(Open)
            return state, end_time, checked_state, open_states

        if tuple(state) in Closed:
            continue

        Closed.add(tuple(state))

        children = generate_children(state, N)

        # dodanie nowego stanu do kolejki, przeszukiwanie w odwrotnej kolejności
        for child in reversed(children):
            if tuple(child) not in Closed and child not in Open:
                # print(f"dfs dodanie dziecka do open: {child}")
                Open.append(child)
        # print(f"Open: {list(Open)}")
        # print(f"Closed: {Closed}\n")

    return [], 0, checked_state, len(Open)


# metoda do testowania kilku wartości N
def run_experiments():
    queens_list = [4, 5, 6, 7, 8]
    results = []
    # zapis do pliku
    output_lines = []

    for N in queens_list:
        print("Exsperymenty dla n-hetmanów\n")

        output_lines.append("Exsperymenty dla n-hetmanów\n")
        # BFS
        bfs_solutions, bfs_time, bfs_closed_state, bfs_open_states = bfs_n_hetman(N)

        # DFS
        dfs_solutions, dfs_time, dfs_closed_state, dfs_open_states = dfs_n_hetman(N)

        # BestFS H1
        bestFS_h1_solutions, bestFS_h1_time, bestFS_h1_close_state, bestFS_h1_open_state, bestFS_h1_val = bestFS_n_hetman(N, heuristic='H1') 

        # BestFS H2
        bestFS_h2_solutions, bestFS_h2_time, bestFS_h2_close_state, bestFS_h2_open_state, bestFS_h2_val = bestFS_n_hetman(N, heuristic='H2')

        # BestFS HDOD
        bestFS_hdod_solutions, bestFS_hdod_time, bestFS_hdod_close_state, bestFS_hdod_open_state, bestFS_hdod_val = bestFS_n_hetman(N, heuristic='HDOD') 
 

        # zapis wyników w results
        results.append({
            "N": N,
            "czas BFS": bfs_time,
            "czas DFS": dfs_time,
            "czas BestFS H1": bestFS_h1_time,
            "czas BestFS H2": bestFS_h2_time,
            "czas BestFS HDOD": bestFS_hdod_time,
            "stan Closed BFS": bfs_closed_state,
            "stan Closed DFS": dfs_closed_state,
            "stan Closed BestFS H1": bestFS_h1_close_state,
            "stan Closed BestFS H2": bestFS_h2_close_state,
            "stan Closed BestFS HDOD": bestFS_hdod_close_state,
            "stan Open BFS": bfs_open_states,
            "stan Open DFS": dfs_open_states,
            "stan Open BestFS H1": bestFS_h1_open_state,
            "stan Open BestFS H2": bestFS_h2_open_state,
            "stan Open BestFS HDOD": bestFS_hdod_open_state,
            "value BestFS HDOD": bestFS_hdod_val,
        })

        # wyniki dla aktualnego N
        print(f"wynik dla N = {N}")
        print(f" BFS - czas: {bfs_time:.8f} s, stany Closed: {bfs_closed_state}, stany Open: {bfs_open_states}")
        print(f" DFS - czas: {dfs_time:.8f} s, stany Closed: {dfs_closed_state}, stany Open: {dfs_open_states}")
        print(f" BestFS H1 - czas: {bestFS_h1_time:.8f} s, stany Closed: {bestFS_h1_close_state}, stany Open: {bestFS_h1_open_state}")
        print(f" BestFS H2 - czas: {bestFS_h2_time:.8f} s, stany Closed: {bestFS_h2_close_state}, stany Open: {bestFS_h2_open_state}")
        print(f" BestFS HDOD - czas: {bestFS_hdod_time:.8f} s, stany Closed: {bestFS_hdod_close_state}, stany Open: {bestFS_hdod_open_state}, wartosc: {bestFS_hdod_val}")
        print('=' * 50)

        output_lines.append(f"wynik dla N = {N}")
        output_lines.append(f" BFS - czas: {bfs_time:.8f} s, stany Closed: {bfs_closed_state}, stany Open: {bfs_open_states}")
        output_lines.append(f" DFS - czas: {dfs_time:.8f} s, stany Closed: {dfs_closed_state}, stany Open: {dfs_open_states}")
        output_lines.append(f" BestFS H1 - czas: {bestFS_h1_time:.8f} s, stany Closed: {bestFS_h1_close_state}, stany Open: {bestFS_h1_open_state}")
        output_lines.append(f" BestFS H2 - czas: {bestFS_h2_time:.8f} s, stany Closed: {bestFS_h2_close_state}, stany Open: {bestFS_h2_open_state}")
        output_lines.append(f" BestFS HDOD - czas: {bestFS_hdod_time:.8f} s, stany Closed: {bestFS_hdod_close_state}, stany Open: {bestFS_hdod_open_state}, wartosc: {bestFS_hdod_val}")
        output_lines.append('=' * 50)

    with open("results_eksperiments.txt", "w") as file:
        file.write("\n".join(output_lines))

    return pd.DataFrame(results)


# ====== rysowanie wykresów =====
def plot_results(df_results):
    fig, axes = plt.subplots(3, 1, figsize=[12, 12])

    # wykers czasów
    axes[0].plot(df_results["N"], df_results["czas BFS"], marker='o', label='BFS')
    axes[0].plot(df_results["N"], df_results["czas DFS"], marker='o', label='DFS')
    axes[0].plot(df_results["N"], df_results["czas BestFS H1"], marker='o', label='H1')
    axes[0].plot(df_results["N"], df_results["czas BestFS H2"], marker='o', label='H2')
    axes[0].plot(df_results["N"], df_results["czas BestFS HDOD"], marker='o', label='HDOD')
    axes[0].set_title("Porównanie czasów")
    axes[0].set_xlabel("Liczba Hetmanów N")
    axes[0].set_ylabel("Czas wykonania")
    axes[0].set_yscale('log')
    axes[0].legend(loc='upper left')
    axes[0].grid(True)

    # wykres 2 stany Closed (sprawdzonych)
    axes[1].plot(df_results["N"], df_results["stan Closed BFS"], marker='o', label='BFS')
    axes[1].plot(df_results["N"], df_results["stan Closed DFS"], marker='o', label='DFS')
    axes[1].plot(df_results["N"], df_results["czas BestFS H1"], marker='o', label='H1')
    axes[1].plot(df_results["N"], df_results["czas BestFS H2"], marker='o', label='H2')
    axes[1].plot(df_results["N"], df_results["czas BestFS HDOD"], marker='o', label='HDOD')
    axes[1].set_title("Porównanie Closed")
    axes[1].set_xlabel("Liczba Hetmanów N")
    axes[1].set_ylabel("Liczba stanów Close")
    axes[1].set_yscale('log')
    axes[1].legend(loc='upper left')
    axes[1].grid(True)

    # wykres 3 stany Closed (sprawdzonych)
    axes[2].plot(df_results["N"], df_results["stan Open BFS"], marker='o', label='BFS')
    axes[2].plot(df_results["N"], df_results["stan Open DFS"], marker='o', label='DFS')
    axes[2].plot(df_results["N"], df_results["czas BestFS H1"], marker='o', label='H1')
    axes[2].plot(df_results["N"], df_results["czas BestFS H2"], marker='o', label='H2')
    axes[2].plot(df_results["N"], df_results["czas BestFS HDOD"], marker='o', label='HDOD')
    axes[2].set_title("Porównanie Open")
    axes[2].set_xlabel("Liczba Hetmanów N")
    axes[2].set_ylabel("Liczba stanów Open")
    axes[2].set_yscale('log')
    axes[2].legend(loc='upper left')
    axes[2].grid(True)


    plt.tight_layout()
    plt.savefig("results_plot.png")
    plt.show()


# ====== Uruchomienie kodu =======
if __name__ == "__main__":
    df_results = run_experiments()
    plot_results(df_results)



"""
WNIOSKI


"""