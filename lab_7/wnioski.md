# Wnioski

Wnioski z przeprowadzonych eksperymentów nad regresją funkcji

$$
f(x_1,x_2)=\cos(x_1*x_2)*\cos(2*x_1)
$$ 

przy pomocy własnej implementacji sieci MLP oraz **sklearn.MLPRegressor**


## Porównanie z sklearn.MLPregressor
MAE - Mean Absolute Error (Średni błąd Bezwzględny)

**MAE własnej MLP**  : &asymp; 0,3838

**MAE 'sklearn'** : &asymp; 0,3799

Prosta implementacja wstecznej propagacji  osiaga porównywalną jakość do gotowych rozwiazań bibliotecznej sieci.


## Wpływ liczby epok
Przy 100 - 1000 epokach MAE jest wysokie &asymp; 0,42

Rosnąc do  7000 - 10000 epok, błąd spaada do  &asymp; 0,38

Po około 7000 epokach dalsze korzyści maleją  - sieć stabilizuje się


## Wpływ współczynnika uczenia (&eta;)
Przy bardzo małym &eta; = 0,0001 -> uczenie jest bardzo powolne (MAE &asymp; 0,42)

Przy dużym &eta; = 0.1 -> uczenie ma najlepsze wyniki (MAE &asymp; 0,316)

Zbyt duży krok (&eta; > 0,1) grozi niestabilnością  (overflow/NaN), dlatego warto zastosować **clipping** aby zabezpieczyć się przed błędem przepełnienia arytmetycznego

**clipping** - ograniczenia skoku gradientu


## Wpływ liczby neuronów w warstwie ukrytej
Sieci z 5 - 20 neuronami uzyskują MAE &asymp; 0,38

Powyżej 50 neuronów nie widać poprawy, czasem się pogarsza z powodu "szumu" w aktualizacjach


## Jakość aproksymacji
Model wyznacza niemal prostą, pochyloną płaszczyznę zamiast falującego orginału

Widać ograniczoną zdolność MLP do uchwycenia nieliniowości przy aktualnej konfiguracji



## Podsumowanie
Samodzielne MLP działa i osiąga wyniki zbliżone do 'sklearn', ale aby precyzyjnie odwzorować powierzchnię funkcji, warto zastosować bardziej zaawansowane techniki uczenia i eksperymentować z architektórą
- funkcji aktywacji np. Relu, tanh
- funkcji optymalizacji np. Adam, RMSprop
- regularyzacja np. L2, dropout
- funkcje straty np. MSE, cross-entropy