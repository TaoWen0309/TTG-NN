import gudhi as gd
import numpy as np

def diagram_from_simplex_tree(st, mode, dim=0):
    st.compute_persistence(min_persistence=-1.)
    dgm0 = st.persistence_intervals_in_dimension(0)[:, 1]

    if mode == "superlevel":
        dgm0 = - dgm0[np.where(np.isfinite(dgm0))]
    elif mode == "sublevel":
        dgm0 = dgm0[np.where(np.isfinite(dgm0))]
    if dim==0:
        return dgm0
    elif dim==1:
        dgm1 = st.persistence_intervals_in_dimension(1)[:,0]
        return dgm0, dgm1

def sum_diag_from_point_cloud(X, mode="superlevel"):
    rc = gd.RipsComplex(points=X)
    st = rc.create_simplex_tree(max_dimension=1)
    dgm = diagram_from_simplex_tree(st, mode=mode)
    sum_dgm = np.sum(dgm)
    return sum_dgm