from collections import deque
import time
import pandas as pd
import matplotlib.pyplot as plt


# metoda sprawzająca bezpieczne pola
def is_safe(queens, x, y):
    for qx, qy in queens:
        # sprawdzenie warunków bicia
        if qx == x or qy == y or abs(qx - x) == abs(qy - y):
            return False
    return True


checked_state = 0


# metoda generujaca potomków
def generate_children(state, N):
    global checked_state
    # index wiersza , do którego dodajemy nowego hetmana -> poziom w drzewie
    x = len(state)
    if x == N:
        return [state]
    children = []
    # unikanie powtórzeń - optymalizacja
    taken_columns = {qy for _, qy in state}

    for y in range(N):
        if y not in taken_columns: #and is_safe(state, x, y):
        #if is_safe(state, x, y):
            # tworzenie nowego stanu
            new_state = state + [(x, y)]
            checked_state += 1
            # print(f"kolejne dziecko {checked_state}: {new_state}")
            children.append(new_state)
    return children


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
        if len(state) == N and is_safe(state):
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
        if len(state) == N and  is_safe(state):

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

        # zapis wyników w results
        results.append({
            "N": N,
            "czas BFS": bfs_time,
            "czas DFS": dfs_time,
            "stan Closed BFS": bfs_closed_state,
            "stan Closed DFS": dfs_closed_state,
            "stan Open BFS": bfs_open_states,
            "stan Open DFS": dfs_open_states,
        })

        # wyniki dla aktualnego N
        print(f"wynik dla N = {N}")
        print(f" BFS - czas: {bfs_time:.8f} s, stany Closed: {bfs_closed_state}, stany Open: {bfs_open_states}")
        print(f" DFS - czas: {dfs_time:.8f} s, stany Closed: {dfs_closed_state}, stany Open: {dfs_open_states}")
        print('=' * 50)

        output_lines.append(f"wynik dla N = {N}")
        output_lines.append(f" BFS - czas: {bfs_time:.8f} s, stany Closed: {bfs_closed_state}, stany Open: {bfs_open_states}")
        output_lines.append(f" DFS - czas: {dfs_time:.8f} s, stany Closed: {dfs_closed_state}, stany Open: {dfs_open_states}")
        output_lines.append('=' * 50)

    with open("wyniki_eksperiments_bez_opt.txt", "w") as file:
        file.write("\n".join(output_lines))

    return pd.DataFrame(results)


# ====== rysowanie wykresów =====
def plot_results(df_results):
    fig, axes = plt.subplots(3, 1, figsize=[12, 12])

    # wykers czasów
    axes[0].plot(df_results["N"], df_results["czas BFS"], marker='o', label='BFS')
    axes[0].plot(df_results["N"], df_results["czas DFS"], marker='o', label='DFS')
    axes[0].set_title("Czas BFS vs DFS")
    axes[0].set_xlabel("Liczba Hetmanów N")
    axes[0].set_ylabel("Czas wykonania")
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True)

    # wykres 2 stany Closed (sprawdzonych)
    axes[1].plot(df_results["N"], df_results["stan Closed BFS"], marker='o', label='BFS')
    axes[1].plot(df_results["N"], df_results["stan Closed DFS"], marker='o', label='DFS')
    axes[1].set_title("Closed BFS vs DFS")
    axes[1].set_xlabel("Liczba Hetmanów N")
    axes[1].set_ylabel("Liczba stanów Close")
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True)

    # wykres 3 stany Closed (sprawdzonych)
    axes[2].plot(df_results["N"], df_results["stan Open BFS"], marker='o', label='BFS')
    axes[2].plot(df_results["N"], df_results["stan Open DFS"], marker='o', label='DFS')
    axes[2].set_title("Open BFS vs DFS")
    axes[2].set_xlabel("Liczba Hetmanów N")
    axes[2].set_ylabel("Liczba stanów Open")
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("wyniki_plot_bez_opt.png")
    plt.show()


# ====== Uruchomienie kodu =======

df_results = run_experiments()
plot_results(df_results)

"""
WNIOSKI:

Algorytm DFS okazał się bardziej wydajny pod wzgledem czasowym niż BFS, szczególnie gdy N rośnie.
Jednak mimo wszystko algorytm BFS lepiej systematyzuje przeszukiwanie, ale zajmuje coraz wiecej pamieci gdy N rośnie.
Dzieki optymalizacji i wyłączeniu już raz odwiedzonych stanów czasy w obu algorytmach się poprawiły,
jest to szczególnie widoczne dla BFS (N=10) czas z 12.70 zmniejszył się do 8.23 (w zalezności od uruchomienia). 
"""
