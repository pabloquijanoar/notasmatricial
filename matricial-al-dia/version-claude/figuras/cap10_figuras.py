"""
Figuras del Capitulo 10 (PCA y clasificacion):
  clasif_pca_mama_digitos.pdf
  clasif_digitos_subespacios.pdf
Ademas imprime todos los numeros que se citan en el texto del capitulo.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_digits, load_wine

from herramientas_cap10_11 import (separar, estandarizar, pca_svd, scores, pve,
                                   centroides, clasificar_centroide, aciertos,
                                   entrenar_subespacios, clasificar_subespacio,
                                   matriz_confusion)

C1, C2, C3 = "#2A78D6", "#EB6834", "#1BAF7A"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})


def centroide_acc(Xtr, ytr, Xte, yte):
    cl, M = centroides(Xtr, ytr)
    return aciertos(yte, clasificar_centroide(Xte, cl, M))


def curva_pca(Xtr, ytr, Xte, yte, ds):
    mu, V, s = pca_svd(Xtr)
    out = []
    for d in ds:
        Ztr, Zte = scores(Xtr, mu, V, d), scores(Xte, mu, V, d)
        out.append(centroide_acc(Ztr, ytr, Zte, yte))
    return np.array(out), pve(s)


# =========================================================== datos ==========
mama = load_breast_cancer()
Xm, ym = mama.data, mama.target
Xmtr, Xmte, ymtr, ymte = separar(Xm, ym)
Xmtr_s, Xmte_s = estandarizar(Xmtr, Xmte)

dig = load_digits()
Xd, yd = dig.data, dig.target
Xdtr, Xdte, ydtr, ydte = separar(Xd, yd)

vino = load_wine()
Xv, yv = vino.data, vino.target
Xvtr, Xvte, yvtr, yvte = separar(Xv, yv)
Xvtr_s, Xvte_s = estandarizar(Xvtr, Xvte)

print("=" * 62)
print("n, p, C :  mama", Xm.shape, len(np.unique(ym)),
      "| digitos", Xd.shape, len(np.unique(yd)),
      "| vino", Xv.shape, len(np.unique(yv)))
print("-" * 62)
print("mama   centroide sin estandarizar : %.3f" % centroide_acc(Xmtr, ymtr, Xmte, ymte))
print("mama   centroide estandarizado    : %.3f" % centroide_acc(Xmtr_s, ymtr, Xmte_s, ymte))
print("vino   centroide sin estandarizar : %.3f" % centroide_acc(Xvtr, yvtr, Xvte, yvte))
print("vino   centroide estandarizado    : %.3f" % centroide_acc(Xvtr_s, yvtr, Xvte_s, yvte))
print("vino   desvios de entrenamiento   : max %.1f, 2do %.1f"
      % tuple(np.sort(Xvtr.std(axis=0, ddof=1))[::-1][:2]))
print("vino   variable de mayor desvio   :", vino.feature_names[int(np.argmax(Xvtr.std(axis=0)))])
print("digitos centroide en R^64         : %.3f" % centroide_acc(Xdtr, ydtr, Xdte, ydte))

# ================================== figura 1: PCA mama + curva digitos ======
accm, pvem = curva_pca(Xmtr_s, ymtr, Xmte_s, ymte, range(1, 31))
dsd = list(range(1, 41))
accd, pved = curva_pca(Xdtr, ydtr, Xdte, ydte, dsd)

print("-" * 62)
for d in (1, 2, 4, 10, 30):
    print("mama    PCA d=%-2d : %.3f   (PVE %.3f)" % (d, accm[d - 1], pvem[d - 1]))
for d in (2, 10, 20, 40):
    print("digitos PCA d=%-2d : %.3f   (PVE %.3f)" % (d, accd[d - 1], pved[d - 1]))

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))

mu, V, s = pca_svd(Xmtr_s)
Z = scores(np.vstack([Xmtr_s, Xmte_s]), mu, V, 2)
yy = np.concatenate([ymtr, ymte])
for k, (c, lab) in enumerate(zip((C2, C1), ("maligno", "benigno"))):
    ax[0].scatter(Z[yy == k, 0], Z[yy == k, 1], s=11, c=c, label=lab,
                  alpha=.75, linewidths=0)
ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
ax[0].set_title("Tumores de mama: PC1 contra PC2\n(variables estandarizadas)", fontsize=9.5)
ax[0].legend(frameon=False, loc="upper right")

ax[1].plot(dsd, accd, "o-", ms=3, c=C1, label="aciertos en prueba")
ax[1].plot(dsd, pved[:len(dsd)], "s--", ms=3, c=C2, label=r"PVE($d$)")
ax[1].axhline(centroide_acc(Xdtr, ydtr, Xdte, ydte), color="gray", lw=.9, ls=":")
ax[1].text(1.5, centroide_acc(Xdtr, ydtr, Xdte, ydte) + .035,
           r"centroide en $\mathbf{R}^{64}$", fontsize=8, color="gray")
ax[1].set_xlabel(r"$d$ (cantidad de componentes)"); ax[1].set_ylim(0, 1.05)
ax[1].set_title("Dígitos: PCA $+$ centroide más cercano", fontsize=9.5)
ax[1].legend(frameon=False, loc="lower right")

fig.tight_layout()
fig.savefig("clasif_pca_mama_digitos.pdf")
plt.close(fig)

# ========================= figura 2: centroides y subespacios en digitos ====
qs = list(range(0, 31))
acc_q_te, acc_q_tr = [], []
for q in qs:
    cl, M, Vs = entrenar_subespacios(Xdtr, ydtr, q)
    acc_q_te.append(aciertos(ydte, clasificar_subespacio(Xdte, cl, M, Vs)))
    acc_q_tr.append(aciertos(ydtr, clasificar_subespacio(Xdtr, cl, M, Vs)))
acc_q_te, acc_q_tr = np.array(acc_q_te), np.array(acc_q_tr)
print("-" * 62)
for q in (0, 5, 10, 15, 20, 25, 30):
    print("digitos subespacio q=%-2d : prueba %.3f  entren. %.3f"
          % (q, acc_q_te[q], acc_q_tr[q]))
print("digitos subespacio: maximo %.3f en q=%d" % (acc_q_te.max(), int(acc_q_te.argmax())))

fig = plt.figure(figsize=(10.5, 3.9))
gs = fig.add_gridspec(2, 5, left=.035, right=.455, top=.80, bottom=.10,
                      wspace=.12, hspace=.28)
cl, M, _ = entrenar_subespacios(Xdtr, ydtr, 0)
for j in range(10):
    a = fig.add_subplot(gs[j // 5, j % 5])
    a.imshow(M[j].reshape(8, 8), cmap="gray_r"); a.set_xticks([]); a.set_yticks([])
    a.grid(False); a.set_title(str(cl[j]), fontsize=8.5, pad=2)
fig.text(.245, .885, "Centroides de las diez clases", fontsize=9.5, ha="center")

ax2 = fig.add_axes([.56, .14, .40, .68])
ax2.plot(qs, acc_q_tr, "s--", ms=3, c=C2, label="entrenamiento")
ax2.plot(qs, acc_q_te, "o-", ms=3.5, c=C1, label="prueba")
ax2.set_xlabel(r"$q$ (dimensión del subespacio de cada clase)")
ax2.set_ylabel("proporción de aciertos")
ax2.set_title("Subespacio más cercano", fontsize=9.5)
ax2.legend(frameon=False, loc="lower right")
fig.savefig("clasif_digitos_subespacios.pdf")
plt.close(fig)

# ---------------------------------------------- matriz de confusion mama ----
cl, M = centroides(Xmtr_s, ymtr)
K = matriz_confusion(ymte, clasificar_centroide(Xmte_s, cl, M), cl)
print("-" * 62)
print("mama, matriz de confusion (prueba, estandarizado):\n", K)
print("=" * 62)
