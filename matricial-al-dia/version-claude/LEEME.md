# Carpeta del apunte — versión al 25 de octubre de 2026

Esta carpeta reemplaza por completo a `2026/notas/version-claude/`.
Es la única versión al día: borrá la vieja y poné ésta en su lugar.

Compila limpio con `latexmk -pdf main.tex`: **268 páginas**, sin referencias
ni citas rotas, sin etiquetas duplicadas.

## Qué estaba desactualizado en tu carpeta

Tu `version-claude` había quedado en el 5 de octubre. Faltaba:

- Todo lo de los zips `matricial-actualizado-1.zip` (6/10) y
  `matricial-actualizado-2.zip` (17/10), que estaban en `Claude outputs/`
  sin descomprimir. De ahí vienen las versiones nuevas de `SVD.tex`,
  `simetricas.tex`, `normas-25-pablo.tex`, `autovalores y autovectores.tex`
  y las figuras del tigre.
- Los capítulos 9, 10 y 11 actualizados por la autora e integrados (19/10).
- El capítulo 12 revisado (25/10).
- El capítulo 13, nuevo (25/10).

## Tres cosas que había cambiado y nunca te mandé

Ya están aplicadas acá:

1. `main.tex`: `\usepackage{booktabs}` (sin eso no compila la tabla PCA/LDA
   del capítulo 11).
2. `main.tex`: `\input{\VA minimoscuadrados}` descomentado, para el cap. 13.
3. `referencias.bib`: entrada `BTLKM2008` (la cita de interfaces
   cerebro-computadora del capítulo 11).

Además regeneré `figuras/clasif_digitos_medias.pdf`, que el capítulo 10 nuevo
necesita y no existía; la genera `figuras/cap10_fig_medias.py`.

## Los capítulos

| # | archivo | estado |
|---|---|---|
| 1 | `cuatrosubespacios.tex` | sin cambios desde el 4/10 |
| 2 | `LU-eliminacion.tex` | sin cambios desde el 4/10 |
| 3 | `ortogonalidad.tex` | del zip del 17/10 |
| 4 | `autovalores y autovectores.tex` | del zip del 17/10 |
| 5 | `schur.tex` | del zip del 17/10 |
| 6 | `simetricas.tex` | del zip del 17/10 (Rayleigh independiente, ejemplos) |
| 7 | `normas-25-pablo.tex` | del zip del 17/10 (confiabilidad, entregas en el cuerpo) |
| 8 | `SVD.tex` | del zip del 17/10 (§8.2 matricial, dos esquemas, tres normas) |
| 9 | `I9_pca.tex` | 19/10, sin entregas |
| 10 | `I10_pca_clasificacion.tex` | 19/10, versión de medias de la autora |
| 11 | `I11_lda.tex` | 19/10, versión corta de la autora |
| 12 | `normas-matrices-25-pablo.tex` | 25/10, 21 ejercicios |
| 13 | `minimoscuadrados.tex` | 25/10, nuevo |

Entregas (ejercicios de desarrollo): viven en los capítulos 1 a 8. Los
capítulos 9 a 13 no tienen; las que se sacaron están guardadas en los docs del
proyecto `claude/entregas-caps-9-11-guardadas.md` y
`claude/entregas-cap12-guardadas.md`.

## Aparte

- `tarea-svd-2026.tex` es el trabajo práctico de la SVD, autocontenido.
  Se compila solo; con `\solucionestrue` salen las soluciones.
- `soluciones-entregas.tex` quedó como estaba (23/9); no lo toqué.
