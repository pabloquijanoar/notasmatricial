"""
Figuras del Capitulo 11 (LDA):
  lda_mama_histogramas.pdf
  lda_vino_pca_lda.pdf
Ademas imprime todos los numeros que se citan en el texto del capitulo.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_digits, load_wine

from herramientas_cap10_11 import (separar, estandarizar, pca_svd, scores,
                                   centroides, clasificar_centroide, aciertos,
                                   covarianza_pooled, direccion_fisher, lda_qr,
                                   columnas_no_constantes, dispersiones)

C1, C2, C3 = "#2A78D6", "#EB6834", "#1BAF7A"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})

# ================================================== mama: Fisher vs PC1 =====
mama = load_breast_cancer()
Xm, ym = mama.data, mama.target          # 0 = maligno, 1 = benigno
Xtr, Xte, ytr, yte = separar(Xm, ym)

v = direccion_fisher(Xtr, ytr)
cl, M = centroides(Xtr, ytr)
umbral = 0.5 * (M @ v).sum()
sentido = 1.0 if (M[0] @ v) > (M[1] @ v) else -1.0
pred = np.where(sentido * (Xte @ v) > sentido * umbral, cl[0], cl[1])
print("=" * 62)
print("mama  LDA (dos clases), aciertos en prueba : %.3f" % aciertos(yte, pred))

mu, V, s = pca_svd(estandarizar(Xtr, Xte)[0])
Xtr_s, Xte_s = estandarizar(Xtr, Xte)
mu, V, s = pca_svd(Xtr_s)
pc1_tr = scores(Xtr_s, mu, V, 1).ravel()
f_tr = Xtr @ v

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.7))
for a, dat, tit in ((ax[0], pc1_tr, "Proyección sobre PC1\n(variables estandarizadas)"),
                    (ax[1], f_tr, r"Proyección sobre la dirección de Fisher $v=S^{-1}(\bar x_1-\bar x_2)$")):
    b = np.linspace(dat.min(), dat.max(), 38)
    a.hist(dat[ytr == 0], bins=b, color=C2, alpha=.72, label="maligno")
    a.hist(dat[ytr == 1], bins=b, color=C1, alpha=.72, label="benigno")
    a.set_title(tit, fontsize=9.5); a.legend(frameon=False)
    a.set_ylabel("cantidad de tumores")
ax[1].axvline(umbral, color="k", lw=1.1, ls="--")
ax[1].annotate("umbral", xy=(umbral, ax[1].get_ylim()[1] * .92),
               xytext=(6, 0), textcoords="offset points", fontsize=8)
fig.tight_layout()
fig.savefig("lda_mama_histogramas.pdf")
plt.close(fig)

# ======================================================= vinos: PCA vs LDA ==
vino = load_wine()
Xv, yv = vino.data, vino.target
Xvtr, Xvte, yvtr, yvte = separar(Xv, yv)
Xvtr_s, Xvte_s = estandarizar(Xvtr, Xvte)

W, lam = lda_qr(Xv, yv)
print("vino  autovalores generalizados de (S_B,S_W)  : %.2f  %.2f" % (lam[0], lam[1]))
Wtr, _ = lda_qr(Xvtr, yvtr)
clv, Mv = centroides(Xvtr, yvtr)
print("vino  LDA (2 direcciones) aciertos en prueba  : %.3f"
      % aciertos(yvte, clasificar_centroide(Xvte @ Wtr, clv, Mv @ Wtr)))
Wtr_s, _ = lda_qr(Xvtr_s, yvtr)
clv_s, Mv_s = centroides(Xvtr_s, yvtr)
print("vino  LDA con variables estandarizadas        : %.3f"
      % aciertos(yvte, clasificar_centroide(Xvte_s @ Wtr_s, clv_s, Mv_s @ Wtr_s)))

# las dos figuras usan los 178 vinos
mu_v, V_v, s_v = pca_svd((Xv - Xv.mean(0)) / Xv.std(0, ddof=1))
Zpca = scores((Xv - Xv.mean(0)) / Xv.std(0, ddof=1), mu_v, V_v, 2)
# normalizadas con la covarianza pooled: W^T S W = I  (S = S_W/(n-C))
W_S = W * np.sqrt(len(Xv) - 3)
Zlda = (Xv - Xv.mean(0)) @ W_S
clA, MA = centroides(Zlda, yv)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
cols = (C1, C2, C3)
for k in range(3):
    ax[0].scatter(Zpca[yv == k, 0], Zpca[yv == k, 1], s=16, c=cols[k],
                  label="cultivar %d" % (k + 1), alpha=.8, linewidths=0)
ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
ax[0].set_title("PCA (variables estandarizadas)", fontsize=9.5)
ax[0].legend(frameon=False, fontsize=8)

# regiones del centroide mas cercano en el plano de LDA
mx = .10 * (Zlda[:, 0].max() - Zlda[:, 0].min())
my = .10 * (Zlda[:, 1].max() - Zlda[:, 1].min())
x0, x1 = Zlda[:, 0].min() - mx, Zlda[:, 0].max() + mx
y0, y1 = Zlda[:, 1].min() - my, Zlda[:, 1].max() + my
gx, gy = np.meshgrid(np.linspace(x0, x1, 420), np.linspace(y0, y1, 420))
G = np.c_[gx.ravel(), gy.ravel()]
R = clasificar_centroide(G, clA, MA).reshape(gx.shape)
ax[1].contourf(gx, gy, R, levels=[-.5, .5, 1.5, 2.5],
               colors=[c + "26" for c in cols])
for k in range(3):
    ax[1].scatter(Zlda[yv == k, 0], Zlda[yv == k, 1], s=16, c=cols[k],
                  label="cultivar %d" % (k + 1), alpha=.85, linewidths=0)
    ax[1].scatter(*MA[k], marker="x", s=85, c="k", linewidths=2.4)
    ax[1].scatter(*MA[k], marker="x", s=70, c=cols[k], linewidths=1.5)
ax[1].set_xlabel(r"$z_1$"); ax[1].set_ylabel(r"$z_2$")
ax[1].set_title("LDA: las dos direcciones discriminantes", fontsize=9.5)
ax[1].legend(frameon=False, fontsize=8)
ax[1].set_xlim(x0, x1); ax[1].set_ylim(y0, y1)
fig.tight_layout()
fig.savefig("lda_vino_pca_lda.pdf")
plt.close(fig)

# =================================================== digitos: LDA en R^9 ====
dig = load_digits()
Xd, yd = dig.data, dig.target
Xdtr, Xdte, ydtr, ydte = separar(Xd, yd)
ok = columnas_no_constantes(Xdtr)
Wd, lamd = lda_qr(Xdtr[:, ok], ydtr)
cld, Md = centroides(Xdtr[:, ok], ydtr)
print("digitos LDA con %d direcciones, aciertos en prueba : %.3f"
      % (Wd.shape[1], aciertos(ydte, clasificar_centroide(Xdte[:, ok] @ Wd, cld, Md @ Wd))))
print("digitos  pixeles constantes en entrenamiento       :", int((~ok).sum()))
print("=" * 62)
