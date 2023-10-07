import numpy as np

X = np.array([
    [1, 0, 3],
    [1, 1, 3],
    [1, 0, 1],
    [1, 1, 1]
], dtype=float)

y = np.array([1, 1, 0, 0])

# Initial values
theta = np.array([0.0, -2.0, 1.0], dtype=float)
lambda_val = 0.07

# Convergence criteria
tolerance = 1e-6 
max_iterations = 100

for iteration in range(max_iterations):
    # Computing sigmoid function
    z = X.dot(theta)
    h = 1 / (1 + np.exp(-z))

    # Comuting gradient
    grad = (1 / len(y)) * X.T.dot(h - y) + (lambda_val / len(y)) * theta
    grad[0] -= (lambda_val / len(y)) * theta[0]  # Don't regularize theta_0

    # Computing Hessian
    S = np.diag(h * (1 - h))
    H = (1 / len(y)) * X.T.dot(S).dot(X) + (lambda_val / len(y)) * np.eye(len(theta))
    H[0, 0] -= (lambda_val / len(y))  # Don't regularize Hessian element H_00

    # Updating theta based on Newton's method
    delta_theta = np.linalg.inv(H).dot(grad)
    theta -= delta_theta

    # Checking for convergence based on the tolerance value and the norm of the gradient
    if np.linalg.norm(delta_theta) < tolerance:
        print(f"Converged after {iteration + 1} iterations.")
        break

    print(f"Iteration {iteration + 1}: theta = {theta}")

print("Final theta:", theta)
