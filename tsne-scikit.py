import matplotlib.pyplot as plt

#Unused but required import for doing 3d projections with plt
import mpl_toolkits.mplot3d
from matplotlib import ticker

from sklearn import datasets, manifold

n_samples = 1500
S_points, S_color = datasets.make_s_curve(n_samples, random_state=0)
print("\nSpoint...\n",S_points)
print("\nS_color....\n",S_color)

def plot_3d(points, points_color, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()

def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=12)
    add_2d_scatter(ax, points, points_color)
    plt.tight_layout()
    plt.savefig("{}.png".format(title[:8]))



def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())


#plot_3d(S_points, S_color, "Original S-curve samples")

#Define algorithms for the manifold learning
#Manifold learning is an approach to non-linear dimensionality
#reduction. Algorithms for this task are based on the idea that the dimensionality of many data sets
#is only artificially high

n_neighbors = 12  # neighborhood which is used to recover the locally linear structure
n_components = 2  # number of coordinates for the manifold

#Locally Linear Embeddings
#Can be though of as a series of local PCA which are globally compared to find the best 
#non-linear embedding. 

params = {
    "n_neighbors": n_neighbors,
    "n_components": n_components,
    "eigen_solver": "auto",
    "random_state": 0,
    }

lle_standard = manifold.LocallyLinearEmbedding(method="standard", **params)
s_standard = lle_standard.fit_transform(S_points)

lle_ltsa = manifold.LocallyLinearEmbedding(method="ltsa", **params)
s_ltsa = lle_ltsa.fit_transform(S_points)

lle_hessian = manifold.LocallyLinearEmbedding(method="hessian", **params)
s_hessian = lle_hessian.fit_transform(S_points)

lle_mod = manifold.LocallyLinearEmbedding(method="modified", **params)
s_mod = lle_mod.fit_transform(S_points)



#Make Plots
fig, axs = plt.subplots(
    nrows=2, ncols=2, figsize=(7,7), facecolor="white", constrained_layout=True
)
fig.suptitle("locally Linear Embedding", size=16)

lle_methods = [
    ("Standard locally linear embedding", s_standard),
    ("Local tangent space alignment", s_ltsa),
    ("Hessian eigenmap", s_hessian),
    ("Modified locally linear embedding", s_mod),
]
for ax,method in zip(axs.flat, lle_methods):
    name, points = method
    add_2d_scatter(ax, points, S_color, name)
#plt.show()

#Isomap Embedding
#Non-linear dimensionality redution via Isometric Mapping seeks a lower-dimensional embedding
#which maintains geodesic distances between all points.

isomap = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components, p=1)
s_isomap = isomap.fit_transform(S_points)

plot_2d(s_isomap, S_color, "Isomap Embedding")

#### MULTIDIMENSIONAL SCALLING
#Seeks a low-dimensional representation of the data in which the distances respect well
#the distances in the original high-dimensional space
md_scaling = manifold.MDS(
    n_components=n_components,
    max_iter=50,
    n_init=4,
    random_state=0,
    normalized_stress=False
)
S_scaling = md_scaling.fit_transform(S_points)
#plot_2d(S_scaling, S_color,"MDS")


###Spectral embedding for non-linear dimensionality reduction
spectral = manifold.SpectralEmbedding(
    n_components=n_components,
    n_neighbors=n_neighbors,
    random_state=42
)
S_pectral = spectral.fit_transform(S_points)
plot_2d(S_pectral, S_color, "Spectral Embedding")


####T-distributed Stochastic Neighbor Embedding
#It converts similarities between data points to joint probabilities and tries to minimize the KLD
#between the joint probabilities of the low-dimensional embedding and high-dimensional data.
#It has a cost function that is not convex. i.e., with different initializations we can get different results
#
n_jobs = 2
t_sne = manifold.TSNE(
    n_components=n_components,
    perplexity=30,
    init = "random",
    n_iter = 250,
    random_state=0,
)
S_t_sne = t_sne.fit_transform(S_points)
plot_2d(S_t_sne, S_color, "T-distributed Stochastic \n Neighbor Embedding")
