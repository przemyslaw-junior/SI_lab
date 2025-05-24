import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator, ClassifierMixin


class GaussianNaiveBayes(BaseEstimator, ClassifierMixin):
    # Naiwny klasyfikator Bayesa z rozkładem normalnym ciągły
    def __init__(self):
        self.class_labels_ = None
        self.class_priors_ = None
        self.mean_ = None
        self.variances_ = None
    
    
    # dopasowanie modelu do danych
    # X - macierz cech, y - wektor etykiet
    def fit(self, X: np.ndarray, y: np.ndarray):
        
        # znajdowanie unikalnych klas
        self.class_labels_ = np.unique(y)
        n_classes_ = len(self.class_labels_)
        n_features_ = X.shape[1]
        
        # priority P(Y=y)
        counts = np.array([np.sum(y == label) for label in self.class_labels_])
        self.class_priors_ = counts / y.shape[0]
        
        # srednie i wariancje cech i klas (ddof=1 - estymator nieobciążony)
        self.mean_ = np.zeros((n_classes_, n_features_))
        self.variances_ = np.zeros((n_classes_, n_features_))
        
        for idx, label in enumerate(self.class_labels_):
            X_class = X[y == label]
            self.mean_[idx, :] = X_class.mean( axis=0)
            self.variances_[idx, :] = X_class.var(axis=0, ddof=1)    
        return self
    
    
    # funkcja pomocnicza do obliczania logarytmu
    def _log_gaussian(self, X: np.ndarray, class_idx: int):        
        mean = self.mean_[class_idx]
        var = self.variances_[class_idx]        
        # składnik normalizacji
        log_norm = -0.5 * np.log(2.0 * np.pi * var)        
        # składnik wykładniczy
        log_exp = - ((X - mean) ** 2) / (2.0 * var) 
        return log_norm + log_exp
    
    
    # funkcja do przewidywania etykiet
    def predict_proba(self, X: np.ndarray):
        m = X.shape[0]
        n_classes_ = len(self.class_labels_)
        log_probs = np.zeros((m, n_classes_))
        
        for idx in range(n_classes_):
            log_prior = np.log(self.class_priors_[idx])
            log_likelihood = np.sum(self._log_gaussian(X, idx), axis=1)
            log_probs[:, idx] = log_prior + log_likelihood
            
        # dla stabilizacji numerycznej wybieramy maksymalną wartość
        log_probs -= np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs
    
        
    # dla każdej próby wybieramy klase o najwyzszym prawdopodobieństwie
    def predict(self, X: np.ndarray):
        proba = self.predict_proba(X)
        class_idx = np.argmax(proba, axis=1)
        return self.class_labels_[class_idx]
    
    
if __name__ =='__main__':    
    # wczytywanie i podział danych
    BASE_DIR = os.path.dirname(__file__)

    wine_path = os.path.join(BASE_DIR,'wine', 'wine.data')
    data_wine = np.genfromtxt(wine_path, delimiter=',')
    
    # znaliza zbioru danych
    print("Liczba próbek: ", data_wine.shape[0])
    print("Liczba cech: ", data_wine.shape[1] - 1)
    print("Liczba klas: ", len(np.unique(data_wine[:, 0])))
    print("Etykiety klas: ", np.unique(data_wine[:, 0]))
    print("Rozkład klas: ", {label: int(np.sum(data_wine[:, 0] == label)) for label in np.unique(data_wine[:, 0])})
    print("Brakujące wartości: ", np.isnan(data_wine).sum(axis=0))
    
    X_wine, y_wine = data_wine[:, 1:], data_wine[:, 0]

    X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(X_wine, y_wine, train_size=0.7, random_state=40, stratify=y_wine)

    # trenowanie i ewluacja
    gnb = GaussianNaiveBayes().fit(X_wine_train, y_wine_train)

    accuracy = gnb.score(X_wine_test, y_wine_test) * 100
    print(f'Własna Accuracy: {accuracy:.2f}%')

    sklearn_gnb = GaussianNB().fit(X_wine_train, y_wine_train)
    sklearn_accuracy = sklearn_gnb.score(X_wine_test, y_wine_test) * 100
    print(f'Sklearn Accuracy: {sklearn_accuracy:.2f}%')