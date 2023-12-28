import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Parameters for two normal distributions
mean1 = [1, 1]
cov1 = [[0.025, 0], [0, 0.025]]  # Covariance matrix for first distribution
# cov1 = [[0.7, 1], [0.5, 0.7]]  # Covariance matrix for first distribution
mean2 = [4.5, 0]
cov2 = [[0.025, 0], [0, 0.025]]  # Covariance matrix for second distribution
# cov2 = [[1.6, 0], [0, 0.7]]  # Covariance matrix for second distribution

# Create a grid of (x, y) coordinates
x = np.linspace(-5, 8, 100)
y = np.linspace(-5, 8, 100)
X, Y = np.meshgrid(x, y)

# Calculate the PDFs for each distribution on the grid
pdf1 = multivariate_normal(mean1, cov1).pdf(np.dstack((X, Y)))
pdf2 = multivariate_normal(mean2, cov2).pdf(np.dstack((X, Y)))

# Plotting the contour of the distributions
plt.figure(figsize=(8, 6))
contour2 = plt.contourf(X, Y, pdf2, cmap="Greys", alpha=0.8)
contour1 = plt.contourf(X, Y, pdf1, cmap="Greys", alpha=0.4)
# plt.colorbar(contour1, ax=plt.gca(), aspect=5)
plt.xlim(-1.5, 7.5)
plt.ylim(-3, 4)
plt.axis("off")

# Save the plot to an image file
plt.savefig(
    "contour_plot_fixed.png",
    bbox_inches="tight",
    pad_inches=0,
)
plt.show()
