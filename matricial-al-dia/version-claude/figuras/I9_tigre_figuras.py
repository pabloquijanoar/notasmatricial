import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
im=Image.open('/tmp/tiger_gs.png').convert('L')
im=im.resize((600,620),Image.LANCZOS)
A=np.asarray(im,float)/255.; m,n=A.shape
Image.fromarray((A*255).astype(np.uint8)).save('/tmp/fig/tigre.png')
U,s,Vt=np.linalg.svd(A,full_matrices=False)
def Ad(d): return (U[:,:d]*s[:d])@Vt[:d]
clip=lambda X: np.clip(X,0,1)
plt.rcParams.update({'font.size':10})
ds=[1,5,20,50,100]
fig,ax=plt.subplots(2,3,figsize=(9,6.3))
ax=ax.ravel(); ax[0].imshow(A,cmap='gray',vmin=0,vmax=1); ax[0].set_title(f'Original ({m}×{n})')
for k,d in enumerate(ds):
    pct=100*d*(m+n+1)/(m*n)
    ax[k+1].imshow(clip(Ad(d)),cmap='gray',vmin=0,vmax=1); ax[k+1].set_title(f'$d={d}$  ({pct:.1f}% de los datos)')
for a in ax: a.axis('off')
plt.tight_layout(); plt.savefig('/tmp/fig/tigre_svd.pdf'); plt.close()
# sigmas vs ruido
rng=np.random.default_rng(7); R=rng.random((m,n)); sR=np.linalg.svd(R,compute_uv=False)
fig,ax=plt.subplots(1,2,figsize=(9,3.4))
K=560; ax[0].semilogy(np.arange(1,K+1),s[:K],label='tigre'); ax[0].semilogy(np.arange(1,K+1),sR[:K],'--',label='ruido uniforme')
ax[0].set_xlabel('$i$'); ax[0].set_ylabel(r'$\sigma_i$ (escala log)'); ax[0].legend(); ax[0].set_title('Valores singulares')
e=np.cumsum(s**2)/np.sum(s**2); eR=np.cumsum(sR**2)/np.sum(sR**2)
ax[1].plot(np.arange(1,len(s)+1),e,label='tigre'); ax[1].set_xlabel('$d$'); ax[1].set_ylabel(r'$\sum_{i\leq d}\sigma_i^2/\|A\|_F^2$'); ax[1].set_xlim(0,150); ax[1].set_title('Fracción de $\\|A\\|_F^2$ capturada')
# sin centrar
plt.tight_layout(); plt.savefig('/tmp/fig/tigre_sigmas.pdf'); plt.close()
# comparación a igual presupuesto
d=50; budget=d*(m+n+1); c=1-np.sqrt(budget/(m*n))
kr=int(round(c*m)); kc=int(round(c*n))
B1=A.copy(); r0=(m-kr)//2; c0=(n-kc)//2; B1[r0:r0+kr,:]=0; B1[:,c0:c0+kc]=0
rows=rng.choice(m,kr,replace=False); cols=rng.choice(n,kc,replace=False)
B2=A.copy(); B2[rows,:]=0; B2[:,cols]=0
keepr=np.setdiff1d(np.arange(m),rows); keepc=np.setdiff1d(np.arange(n),cols)
nr=keepr[np.abs(np.arange(m)[:,None]-keepr[None,:]).argmin(1)]
nc=keepc[np.abs(np.arange(n)[:,None]-keepc[None,:]).argmin(1)]
B3=A[np.ix_(nr,nc)]
nF=np.linalg.norm(A,'fro'); rel=lambda X: np.linalg.norm(A-X,'fro')/nF
fig,ax=plt.subplots(1,4,figsize=(11,3.5))
for a,X,t in zip(ax,[B1,B2,B3,clip(Ad(d))],['filas/columnas centrales\npintadas de negro','filas/columnas aleatorias\npintadas de negro','aleatorias, rellenadas con\nla vecina conservada más cercana',f'SVD truncada, $d={d}$']):
    a.imshow(X,cmap='gray',vmin=0,vmax=1); a.set_title(t+f'\nerror rel. Frobenius: {rel(X if X is not B3 else B3):.2f}',fontsize=9); a.axis('off')
plt.tight_layout(); plt.savefig('/tmp/fig/tigre_comparacion.pdf'); plt.close()
print('rels',rel(B1),rel(B2),rel(B3),rel(Ad(d)), 'op', [np.linalg.norm(A-X,2) for X in (B1,B2,B3,Ad(d))])
print('m,n',m,n,'rank',np.linalg.matrix_rank(A),'budget',budget,'c',c,kr,kc)
for dd in [5,20,50,100]:
    E=A-Ad(dd)
    print(dd, 'op',np.linalg.norm(E,2),'s_d+1',s[dd],'F',np.linalg.norm(E,'fro'),'sqrt',np.sqrt(np.sum(s[dd:]**2)),'nuc',np.linalg.norm(E,'nuc'),'sum',np.sum(s[dd:]),'relF',np.linalg.norm(E,'fro')/np.linalg.norm(A,'fro'), 'energy',e[dd-1])
# deletion (painted) vs svd', np.linalg.norm(A-B2,'fro'),np.linalg.norm(A-B1,'fro'),np.linalg.norm(A-Ad(d),'fro'))
print('s1..5',s[:5], 's1^2/total',s[0]**2/np.sum(s**2), 'mean',A.mean())
print('random s', sR[:3], 'energy random 50', eR[49], 'energy random 1', eR[0])
