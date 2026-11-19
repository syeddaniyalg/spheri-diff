import torch

EPSILON = torch.finfo(torch.float64).eps
SOLVERS = ("cg", "gmres")


def dot(first, second):
    return (first * second).sum(dim=-1)


def conjugate_gradient(system, rhs, iterations):
    target = rhs
    residual = rhs.double()
    solution = torch.zeros_like(residual)
    direction = residual
    energy = dot(residual, residual)
    for _ in range(iterations):
        image = system(direction.to(target)).double()
        step = energy / dot(direction, image).clamp_min(EPSILON)
        solution = solution + step[..., None] * direction
        residual = residual - step[..., None] * image
        new_energy = dot(residual, residual)
        direction = residual + (new_energy / energy.clamp_min(EPSILON))[..., None] * direction
        energy = new_energy
    return solution.to(target)


def normalise(vector):
    norm = torch.linalg.vector_norm(vector, dim=-1)
    return vector / norm[..., None].clamp_min(EPSILON), norm


def givens(first, second):
    scale = torch.sqrt(first * first + second * second).clamp_min(EPSILON)
    return first / scale, -second / scale


def arnoldi_column(system, basis, hessenberg, index, target):
    vector = system(basis[index].to(target)).double()
    for row in range(index + 1):
        hessenberg[row][index] = dot(vector, basis[row])
        vector = vector - hessenberg[row][index][..., None] * basis[row]
    basis[index + 1], hessenberg[index + 1][index] = normalise(vector)


def rotate_column(hessenberg, residuals, cosines, sines, index):
    for row in range(index):
        kept = cosines[row] * hessenberg[row][index] - sines[row] * hessenberg[row + 1][index]
        hessenberg[row + 1][index] = cosines[row] * hessenberg[row + 1][index] + sines[row] * hessenberg[row][index]
        hessenberg[row][index] = kept
    cosines[index], sines[index] = givens(hessenberg[index][index], hessenberg[index + 1][index])
    hessenberg[index][index] = (
        cosines[index] * hessenberg[index][index] - sines[index] * hessenberg[index + 1][index]
    )
    residuals[index + 1] = sines[index] * residuals[index]
    residuals[index] = cosines[index] * residuals[index]
    for row in range(index + 1, len(hessenberg)):
        hessenberg[row][index] = torch.zeros_like(hessenberg[index][index])


def generalised_minimal_residual(system, rhs, iterations):
    target = rhs
    basis = [None] * (iterations + 1)
    residuals = [None] * (iterations + 1)
    hessenberg = [[None] * iterations for _ in range(iterations + 1)]
    cosines, sines = [None] * iterations, [None] * iterations
    basis[0], residuals[0] = normalise(rhs.double())
    for index in range(iterations):
        arnoldi_column(system, basis, hessenberg, index, target)
        rotate_column(hessenberg, residuals, cosines, sines, index)
    vectors = torch.stack(basis[:-1], dim=-2)
    residual = torch.stack(residuals[:-1], dim=-1)
    matrix = torch.stack([torch.stack(row, dim=-1) for row in hessenberg[:-1]], dim=-2)
    identity = torch.eye(iterations, dtype=matrix.dtype, device=matrix.device)
    weights = torch.linalg.solve_triangular(matrix + EPSILON * identity, residual[..., None], upper=True)
    return torch.einsum("...ij,...i", vectors, weights.squeeze(-1)).to(target)


def solve_linear(system, rhs, iterations, solver):
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver '{solver}'")
    shape = rhs.shape
    flat = rhs.reshape(shape[0], -1)

    def flat_system(vector):
        return system(vector.reshape(shape)).reshape(shape[0], -1)

    method = conjugate_gradient if solver == "cg" else generalised_minimal_residual
    return method(flat_system, flat, iterations).reshape(shape)
