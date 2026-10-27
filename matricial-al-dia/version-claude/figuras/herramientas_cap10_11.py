"""
Funciones auxiliares para los capitulos 10 (PCA y clasificacion) y 11 (LDA).
Todo con numpy: centroides, PCA por SVD, clasificadores y LDA.
"""
import numpy as np


# ---------------------------------------------------------------- datos ----
def separar(X, y, prop_prueba=0.3, semilla=0):
    """Separacion estratificada entrenamiento/prueba (misma que sklearn)."""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=prop_prueba,
                            random_state=semilla, stratify=y)


def columnas_no_constantes(X, tol=1e-12):
    return X.std(axis=0) > tol


def estandarizar(Xtr, Xte):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0, ddof=1)
    sd[sd < 1e-12] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


# ------------------------------------------------------------------ PCA ----
def pca_svd(Xtr):
    """Devuelve la media, V (p x p) y los valores singulares de Xtr centrada."""
    mu = Xtr.mean(axis=0)
    Xc = Xtr - mu
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt.T, s


def scores(X, mu, V, d):
    return (X - mu) @ V[:, :d]


def pve(s):
    e = s ** 2
    return np.cumsum(e) / e.sum()


# --------------------------------------------------- centroide mas cercano -
def centroides(X, y):
    clases = np.unique(y)
    return clases, np.array([X[y == k].mean(axis=0) for k in clases])


def clasificar_centroide(Xnuevo, clases, M):
    # delta_k(x) = x^T mbar_k - 1/2 |mbar_k|^2 ; se asigna el maximo
    D = Xnuevo @ M.T - 0.5 * (M ** 2).sum(axis=1)
    return clases[np.argmax(D, axis=1)]


def aciertos(y, yhat):
    return float(np.mean(y == yhat))


def matriz_confusion(y, yhat, clases):
    C = len(clases)
    K = np.zeros((C, C), dtype=int)
    for j, kj in enumerate(clases):
        for k, kk in enumerate(clases):
            K[j, k] = np.sum((y == kj) & (yhat == kk))
    return K


# -------------------------------------------- subespacio mas cercano (PCA) -
def entrenar_subespacios(X, y, q):
    clases = np.unique(y)
    M, Vs = [], []
    for k in clases:
        Xk = X[y == k]
        mk = Xk.mean(axis=0)
        M.append(mk)
        if q == 0:
            Vs.append(np.zeros((X.shape[1], 0)))
        else:
            _, _, Vt = np.linalg.svd(Xk - mk, full_matrices=False)
            Vs.append(Vt[:q].T)
    return clases, np.array(M), Vs


def clasificar_subespacio(Xnuevo, clases, M, Vs):
    R = np.empty((Xnuevo.shape[0], len(clases)))
    for k in range(len(clases)):
        U = Xnuevo - M[k]
        R[:, k] = (U ** 2).sum(axis=1) - ((U @ Vs[k]) ** 2).sum(axis=1)
    return clases[np.argmin(R, axis=1)]


# ------------------------------------------------------------- dispersion --
def dispersiones(X, y):
    """S_W, S_B, S_T sin dividir por nada (son X_W^T X_W, etc.)."""
    xbar = X.mean(axis=0)
    p = X.shape[1]
    SW = np.zeros((p, p))
    SB = np.zeros((p, p))
    for k in np.unique(y):
        Xk = X[y == k]
        mk = Xk.mean(axis=0)
        SW += (Xk - mk).T @ (Xk - mk)
        d = (mk - xbar).reshape(-1, 1)
        SB += len(Xk) * (d @ d.T)
    ST = (X - xbar).T @ (X - xbar)
    return SW, SB, ST


# ------------------------------------------------------------------- LDA ---
def covarianza_pooled(X, y):
    SW, _, _ = dispersiones(X, y)
    return SW / (len(X) - len(np.unique(y)))


def direccion_fisher(X, y):
    """Dos clases: v = S^{-1}(xbar_1 - xbar_2) con S la covarianza pooled."""
    clases, M = centroides(X, y)
    assert len(clases) == 2
    S = covarianza_pooled(X, y)
    return np.linalg.solve(S, M[0] - M[1])


def lda_qr(X, y, eps=0.0):
    """LDA por QR + SVD.  Devuelve (W, lambdas) con W^T S_W W = I."""
    clases, M = centroides(X, y)
    xbar = X.mean(axis=0)
    nk = np.array([np.sum(y == k) for k in clases], dtype=float)
    XW = X - M[np.searchsorted(clases, y)]
    p = X.shape[1]
    R = np.linalg.qr(XW, mode='r')
    if eps > 0:
        R = np.linalg.cholesky(R.T @ R + eps * np.eye(p)).T
    Bt = np.sqrt(nk)[:, None] * (M - xbar)
    Z = np.linalg.solve(R.T, Bt.T).T          # Bt R^{-1}
    Ub, sb, Vbt = np.linalg.svd(Z, full_matrices=False)
    m = min(len(clases) - 1, p)
    W = np.linalg.solve(R, Vbt[:m].T)
    return W, sb[:m] ** 2


def clasificar_lda(Xnuevo, X, y, W):
    clases, M = centroides(X, y)
    return clasificar_centroide(Xnuevo @ W, clases, M @ W)


def dejar_uno_afuera(X, y, clasificador):
    ok = 0
    for i in range(len(X)):
        m = np.ones(len(X), dtype=bool)
        m[i] = False
        ok += int(clasificador(X[m], y[m], X[i:i + 1])[0] == y[i])
    return ok / len(X)
