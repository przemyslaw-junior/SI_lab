# metoda sprawdzająca czy hetman jest bezpieczny
def is_safe(queens, x, y):
    for qx, qy in queens:
        # sprawdzenie warunków bicia
        if qx == x or qy == y or abs(qx - x) == abs(qy - y):
            return False
    return True


# metoda rozwiązująca n_hetmanów i zwracająca listę rozwiazań
def n_hetman_solution(N):
    # zbiór współrzędnych hetmanów
    queens = set()
    # lista rozwiązań
    solutions = []

    # rekurencyjna funkcja sledząca
    def backtrack(x):
        # jeżeli dojdziemy do końca -> ilość hetmanów
        if x == N:
            solutions.append(queens.copy())
            return

        # iterowanie po kolumnach w aktualnym wierszu
        for y in range(N):
            # sprawdzenie czy można umieścić hetmana
            if is_safe(queens, x, y):
                # dodnanie hetmana na (x,y)
                queens.add((x, y))
                # przejscie do kolejnego wiersza
                backtrack(x + 1)
                # cofnięcie decyzji
                queens.remove((x, y))

    # startujemy od pierwszego wiersza
    backtrack(0)
    return solutions


# metoda wyświetlająca wyniki
def print_solutions(solutions, N):
    for sol_num, sol in enumerate(solutions):
        print(f"Rozwiazanie {sol_num + 1}:")
        # tworzenie pustej tablicy
        board = [['.'] * N for _ in range(N)]
        # ustawienie hetmanów na pozycji (x,y)
        for x, y in sol:
            board[x][y] = 'Q'

        for row in board:
            print(' '.join(row))
        print("\n")


# ====== Uruchomienie kodu =======
N = 4
solutions = n_hetman_solution(N)

print(f"Liczba rozwiazań dla {N} - hetmanów: {len(solutions)}")
print_solutions(solutions, N)