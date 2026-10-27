"""Figura clasif_digitos_medias.pdf del Capitulo 10 (version de medias)."""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from herramientas_cap10_11 import separar, centroides

C1, C2 = "#2A78D6", "#EB6834"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

dig = load_digits()
Xtr, Xte, ytr, yte = separar(dig.data, dig.target)
clases, M = centroides(Xtr, ytr)
i = 41                                   # un tres del conjunto de prueba
d = np.linalg.norm(Xte[i] - M, axis=1)
print("n_entren =", len(Xtr), "| clase verdadera =", yte[i],
      "| asignada =", int(clases[np.argmin(d)]))
print("distancias:", np.round(d, 2))

fig = plt.figure(figsize=(10.6, 3.5))
gs = fig.add_gridspec(2, 5, left=.025, right=.40, top=.80, bottom=.08,
                      wspace=.10, hspace=.30)
for j in range(10):
    a = fig.add_subplot(gs[j // 5, j % 5])
    a.imshow(M[j].reshape(8, 8), cmap="gray_r")
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
    a.set_title(str(int(clases[j])), fontsize=8.5, pad=2)
fig.text(.212, .885, "Media de cada clase", fontsize=9.5, ha="center")

a = fig.add_axes([.455, .14, .145, .60])
a.imshow(Xte[i].reshape(8, 8), cmap="gray_r")
a.set_xticks([]); a.set_yticks([]); a.grid(False)
a.set_title("Dígito nuevo", fontsize=9.5, pad=6)

a2 = fig.add_axes([.675, .17, .30, .60])
col = [C2 if k == int(np.argmin(d)) else C1 for k in range(10)]
a2.bar(range(10), d, color=col, width=.72)
a2.set_xticks(range(10)); a2.set_xlabel("clase")
a2.set_ylabel(r"$\|x-\bar x_k\|$")
a2.set_title("Distancia a cada media", fontsize=9.5, pad=6)
a2.grid(axis="y", alpha=.25)
a2.annotate("%.1f" % d.min(), (int(np.argmin(d)), d.min()),
            textcoords="offset points", xytext=(0, 3), ha="center",
            fontsize=8, color=C2)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "clasif_digitos_medias.pdf"))
