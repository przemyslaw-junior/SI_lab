
win = +100 # wygrana dla max
loss = -100 # wygrana dla min

node_count = 0 # licznik węzłów
node_order = [] # kolejność przeszukiwania węzłów
prune_events = [] # przycinanie zdarzeń


# funkcja zapisująca informacje o odwiedzonych wezłach
def log_node(label, depth, alpha, beta):
    indent = "  " *depth
    log_line = f"{indent}{label} (alpha={alpha}, beta={beta})"
    node_order.append(log_line)
    #print(log_line)
    

# funkcja dla MAX licząca wartość węzła
def alpha_beta_max(state, alpha, beta, dephth):
    global node_count
    node_count += 1
    label = f"MAX (state={state})"
    
    log_node(label, dephth, alpha, beta)
    
    # warunek terminalny -> ostatnia moneta
    if state ==1:
        result = loss
        node_order.append(" " * (dephth + 1) + f"Terminal: state=1, zwracam {result}")
        return result
    
    best_value = -100 # poczatkowo najgorsza wartość dla max
    
    for move in range(1, min(3, state) + 1):
        if move == state: # przegrana mam ostatnia moneta
            value = loss
            node_order.append(" " * (dephth + 1) + f"Ruch {move}: pobieram wszystkie monety -> wartość {value}")
        elif state - move == 1:
            value = win
            node_order.append(" " * (dephth + 1) + f"Ruch {move}: zostaje jedna moneta -> wartość {value}")
        else:
            node_order.append(" " * (dephth + 1) + f"Ruch {move}: stan {state - move}")
            value = alpha_beta_min(state - move, alpha, beta, dephth + 1)
            node_order.append(" " * (dephth + 1) + f"Ruch {move}: wartość {value}")
            
        # aktualizacja najlepszej wartości
        if value > best_value:
            best_value = value
            
        old_alpha = alpha # aktualizacja alpha tylko w wezłach MAX
        alpha = max(alpha, value)
        node_order.append(" " * (dephth + 1) + f"Ruch {move}: aktualizacja alpha = max: ({old_alpha}, {value}) -> {alpha}")
            
        # sprawdzenie przycięcia
        if alpha >= beta:
            prune_events.append("Przyciecie w " + label + f"po ruchu {move} (alpha={alpha}, beta={beta})")
            node_order.append(" " * (dephth + 1) + f"Przyciecie kolejnego ruchu {move}: ponieważ alpha = {alpha} >= beta = {beta}")
            break
    
    node_order.append(" " * dephth + f"{label} zwracam {best_value}")
    return best_value


# funkcja dla MIN licząca wartość węzła
def alpha_beta_min(state, alpha, beta, depth):
    global node_count
    node_count += 1
    label = f"MIN (state={state})"
    
    log_node(label, depth, alpha, beta)
    
    # warunek terminalny -> ostatnia moneta
    if state == 1:
        result = win
        node_order.append(" " * (depth + 1) + f"Terminal: state=1, zwracam {result}")
        return result
    
    best_value = 100 # poczatkowo najgorsza wartość dla min
    
    for move in range(1, min(3, state) + 1):
        if move == state: # przegrana mam ostatnia moneta
            value = win
            node_order.append(" " * (depth + 1) + f"Ruch {move}: pobieram wszystkie monety -> wartość {value}")
        elif state - move == 1:
            value = loss
            node_order.append(" " * (depth + 1) + f"Ruch {move}: zostaje jedna moneta -> wartość {value}")
        else:
            node_order.append(" " * (depth + 1) + f"Ruch {move}: stan {state - move}")
            value = alpha_beta_max(state - move, alpha, beta, depth + 1)
            node_order.append(" " * (depth + 1) + f"Ruch {move}: wartość {value}")
            
        # aktualizacja najlepszej wartości
        if value < best_value:
            best_value = value
            
        old_beta = beta # aktualizacja beta tylko w wezłach MIN
        beta = min(beta, value)
        node_order.append(" " * (depth + 1) + f"Ruch {move}: aktualizacja beta = min: ({old_beta}, {value}) -> {beta}")
            
        # sprawdzenie przycięcia
        if alpha >= beta:
            prune_events.append("Przyciecie w " + label + f"po ruchu {move} (alpha={alpha}, beta={beta})")
            node_order.append(" " * (depth + 1) + f"Przyciecie kolejnego ruchu {move}: ponieważ alpha = {alpha} >= beta = {beta}")
            break
    
    node_order.append(" " * depth + f"{label} zwracam {best_value}")
    return best_value


def main():
    global node_count, node_order, prune_events
    node_count = 0
    node_order = []
    prune_events = []
    
    initial_state = 5 # początkowy stan gry
    
    print("Stan początkowy:", initial_state)
    
    alpha = -100 # początkowa wartość alpha
    beta = 100 # początkowa wartość beta
    
    move_evaluation = {} # słownik: klucz = ruch, wartość = ocena ruchu
    node_order.append(f"Stan początkowy: {initial_state}")
    
    # dla kazdego mozliwego ruchu gracza A (max)
    for move in range(1, min(3, initial_state) + 1):
        if move == initial_state:
            value = loss
            node_order.append(f"Ruch {move} w korzeniu: pobieram wszystkie monety -> wartość {value}")
        elif initial_state - move == 1:
            value = win
            node_order.append(f"Ruch {move} w korzeniu: zostaje jedna moneta -> wartość {value}")
        else:
            node_order.append(f"Ruch {move} w korzeniu: stan {initial_state - move}")
            value = alpha_beta_min(initial_state - move, alpha, beta, depth=1)
            node_order.append(f"Ruch {move} w korzeniu: wartość {value}")

        move_evaluation[move] = value
        # aktualizacja globalna alpha na poziomie korzenia
        alpha = max(alpha, value)
        
        
    # wyznaczenie najlepszego ruchu
    best_move = max(move_evaluation, key=move_evaluation.get)
    best_value = move_evaluation[best_move]
    
    # podsumowanie wyników
    print("wynik analizy algorytmu alpha-beta dla gry w monety:")
    print("====================================================")
    print("wizualizacja drzewa przeszukiwania (kolejność wezłów i zmian alpha/beta):")
    for line in node_order:
        print(line)
    print("====================================================")
    
    print("zdarzenia przycinania:")
    if prune_events:
        for event in prune_events:
            print(" - " + event)
    else:
        print(" - brak przycieć")
    
    print("====================================================")
    print("liczba odwiedzonych wezłów:", node_count)
    print("ocena ruch w korzeniu:")
    for move, value in move_evaluation.items():
        print(f" - ruch {move} (pobranie {move} monety): {value}")
    print("====================================================")
    
    # uzasadnienie wyboru
    if best_value == win:
        print(f"Najlepszy ruch gracza A to pobranie: {best_move} monety, co gwarantuje wygraną")
    else:
        print(f"żaden ruch nie gwarantuje zwyciestwa, najlepszy ruch to pobranie: {best_move} monety,"
              "który minimalizuje porazkę. Pod warunkiem nierozsadnej gry gracza B.")
  
# uruchomienie programu      
if __name__ == "__main__":
    main()
    