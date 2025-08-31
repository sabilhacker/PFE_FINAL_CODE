from functools import partial
from qiskit_algorithms.optimizers import SPSA, ADAM, NFT, GradientDescent, L_BFGS_B
# def get_optimizer_options(optimizer_type):
#     if optimizer_type == "SPSA":
#         optimizer = partial(SPSA, maxiter=300, learning_rate=0.002, perturbation=0.05)
#         callback = "SPSACallback"
#     elif optimizer_type == "ADAM":
#         optimizer = partial(ADAM, maxiter=600, tol=1e-08, lr=0.002)
#         callback = None
#     elif optimizer_type == "GradientDescent":
#         optimizer = partial(
#             GradientDescent,
#             maxiter=400,
#             learning_rate=0.002,
#             tol=1e-08,
#             perturbation=None,
#         )
#         callback = "SPSACallback"
#     elif optimizer_type == "NFT":
#         optimizer = partial(NFT, disp=False)
#         callback = None
#     elif optimizer_type == "L_BFGS_B":
#         optimizer = partial(
#             L_BFGS_B,
#             maxfun=15000,
#             maxiter=15000,
#             ftol=2.220446049250313e-15,
#             iprint=-1,
#             eps=1e-08,
#         )
#         callback = None
#     return callback, optimizer
def get_optimizer_options(optimizer_type, maxiter, learning_rate, perturbation):
    if optimizer_type == "SPSA":
        optimizer = partial(SPSA, maxiter=maxiter, learning_rate=learning_rate, perturbation=perturbation)
        callback = "SPSACallback"
    return callback, optimizer
