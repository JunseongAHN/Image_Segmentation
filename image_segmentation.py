"""Spectral image segmentation using a graph cut on pixel neighborhoods."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csc_matrix, csgraph, diags, lil_matrix
from scipy.sparse import linalg as sparse_linalg


def get_neighbors(index: int, radius: float, height: int, width: int):
    """Return flattened neighbor indices and distances within radius."""
    row, col = index // width, index % width
    r = int(radius)

    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    distances = np.sqrt((X - col) ** 2 + (Y - row) ** 2)
    mask = distances < radius
    return (X[mask] + Y[mask] * width).astype(int), distances[mask]


class ImageSegmenter:
    """Graph-based image segmenter using brightness and spatial proximity."""

    def __init__(self, filename: str):
        image = imageio.imread(filename)
        self.image = image.astype(np.float32) / 255.0

        if self.image.ndim == 3:
            self.brightness = self.image.mean(axis=2)
        else:
            self.brightness = self.image

        self.shape = self.brightness.shape
        self.flat_brightness = self.brightness.ravel()

    def adjacency(self, radius=5.0, sigma_b=0.05, sigma_x=5.0):
        """Build sparse adjacency matrix and degree vector."""
        height, width = self.shape
        num_pixels = height * width
        A = lil_matrix((num_pixels, num_pixels), dtype=np.float32)

        start = time.time()
        for i in range(num_pixels):
            neighbors, distances = get_neighbors(i, radius, height, width)
            brightness_diff = np.abs(self.flat_brightness[i] - self.flat_brightness[neighbors])
            weights = np.exp(-(brightness_diff / sigma_b) - (distances / sigma_x))
            A[i, neighbors] = weights

        A = csc_matrix(A)
        degree = np.asarray(A.sum(axis=0)).ravel()
        print(f"Built adjacency matrix in {time.time() - start:.2f}s")
        return A, degree

    def cut(self, A, degree):
        """Compute a binary mask from the second smallest eigenvector."""
        L = csgraph.laplacian(A)
        inv_sqrt_degree = np.power(np.maximum(degree, 1e-8), -0.5)
        D_inv_sqrt = diags(inv_sqrt_degree)
        normalized_L = D_inv_sqrt @ L @ D_inv_sqrt

        eigenvalues, eigenvectors = sparse_linalg.eigsh(normalized_L, k=2, which="SM")
        fiedler_vector = eigenvectors[:, np.argsort(eigenvalues)[1]]
        mask = fiedler_vector.reshape(self.shape) > 0

        return mask

    def segment(self, radius=5.0, sigma_b=0.05, sigma_x=5.0):
        A, degree = self.adjacency(radius=radius, sigma_b=sigma_b, sigma_x=sigma_x)
        return self.cut(A, degree)

    def save_result(self, mask, output_path: str):
        """Save original image, positive mask, and negative mask as one figure."""
        if self.image.ndim == 3:
            mask_3c = np.dstack([mask] * 3)
        else:
            mask_3c = mask

        fig, axes = plt.subplots(1, 3, figsize=(10, 4), constrained_layout=True)
        axes[0].imshow(self.image, cmap=None if self.image.ndim == 3 else "gray")
        axes[0].set_title("Input")
        axes[1].imshow(self.image * mask_3c, cmap=None if self.image.ndim == 3 else "gray")
        axes[1].set_title("Segment A")
        axes[2].imshow(self.image * (~mask if self.image.ndim == 2 else ~mask_3c), cmap=None if self.image.ndim == 3 else "gray")
        axes[2].set_title("Segment B")

        for ax in axes:
            ax.axis("off")

        fig.savefig(output_path, dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Spectral image segmentation demo")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output", default="outputs/result.png", help="Path to output visualization")
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument("--sigma-b", type=float, default=0.08)
    parser.add_argument("--sigma-x", type=float, default=5.0)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    segmenter = ImageSegmenter(args.image)
    mask = segmenter.segment(radius=args.radius, sigma_b=args.sigma_b, sigma_x=args.sigma_x)
    segmenter.save_result(mask, args.output)
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
