
# image_segmentation.py


import numpy as np
from scipy import linalg as la

from scipy.sparse import linalg
from imageio import imread
from matplotlib import pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import csgraph
from scipy.sparse import diags
import matplotlib.animation as animation
import time
import cv2




def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    L = laplacian(A)
    eig, eig_vec = la.eig(L)
    #decreasing order
    eig.sort()
    for i in range(len(eig)):
        if (i > tol):
            break
    return i, np.real(eig[1]) #second biggest



# Read a (very) small image.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]



# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        image = imread(filename)
        self.scaled = image / 255
        #if it is color
        if(len(image.shape) == 3):
            self.brightness = self.scaled.mean(axis=2)
        else:
            #if it is not color
            self.brightness = self.scaled

        #get shape of brightness
        self.shape = self.brightness.shape
        #get flat brightness
        self.brightness = np.ravel(self.brightness)


    def show_original(self):
        """Display the original image."""
        #if it is color
        if(len(self.scaled.shape) == 3):
            plt.imshow(self.scaled)
        else:
        #if it is not color
            plt.imshow(self.scaled,cmap = 'gray')

    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""

        #helper function to calculate weights for vertex i
        def get_weights(neighbors, distances, sigma_B2, sigmaX2):
            b_i = self.brightness[i]

            #calculate exponent
            dist = -distances / sigma_X2
            b_j = np.array([self.brightness[j] for j in neighbors])
            b = -abs(b_i - b_j) / sigma_B2

            #calculate weights
            weights = pow(np.e, b + dist)
            return weights

        #calculate m * n
        m, n = self.shape

        mn = m * n

        #initialize sparse matrix
        A = lil_matrix((mn, mn))

        print("get_neighbors...")
        start = time.time()
        for i in range(mn):
            #get neighbors and distances

            neighbors, dist = get_neighbors(i, r, m,n)
            
            #calculate weights
            weights = get_weights(neighbors, dist, sigma_B2, sigma_X2)
            A[i, neighbors] = weights
        print("done... ",time.time()-start," sec")
        #convert format
        A = csc_matrix(A)

        #compute D (sum of each column of A)
        D = np.array(A.sum(axis=0))[0, :]

        return A, D

    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        #compute laplacian
        print("Calculate laplacian...")
        a = time.time()
        L = csgraph.laplacian(A)
        print("Done... " ,time.time()-a," sec")
        

        #compute D^(-1/2)
        diag = np.power(D, -1/2)
        D = diags(diag)
        #construct D-1/2 L D-1/2
        M = D @ L @ D
        #M = lil_matrix(M)
        M = M.toarray()

        #find second smallest eigenvector
        #eigs, vecs = linalg.eigsh(M, which="SM", k=2)
        print("eigenvalues...")
        a = time.time()
        eigs, vecs = linalg.eigs(M, which="SM", k=2)
        eig, vec = eigs[1], vecs[:, 1]
        print(np.sum(vec>0), np.sum(vec<0))
        print("Done... " ,time.time()-a," sec")



        #reshape as m*n matrix
        vec = np.reshape(vec, (self.shape[0], self.shape[1]))

        #construct mask for color images
        mask = vec > 0
        #when all of thing is similar color, we threshhold to all zero
        #I need more careful analysis
        if (la.norm(vec[vec>0], ord=np.inf) - la.norm(vec[vec<0], ord=np.inf) < 1e-6):
            print("here, it didn't spilit")
            return np.zeros(shape=mask.shape, dtype=bool)
        else:
            return mask

    def segment(self, r=5., sigma_B=0.05, sigma_X=5):
        """Display the original image and its segments."""
        ims = []
        A, D = self.adjacency(r,sigma_B,sigma_X)
        #get mask
        mask = self.cut(A,D)
        if self.scaled.shape[-1] == 3:
            inv_mask = np.dstack((~mask,~mask,~mask))
            color_mask = np.dstack((mask,mask,mask))
            #for color
            fig, axs = plt.subplots(1, 3, constrained_layout=True)
            plt.figtext(.5,.9,'Image Segmentation', fontsize=18, ha='center')
            plt.figtext(.5,.85,'Hyperparemeters: r=%.3f, sigma_B=%.3f, sigma_X=%.3f'% (r, sigma_B, sigma_X),fontsize=10,ha='center')
            plt.figtext(.5,.80,'Ratio %d : %d (%.03f)'% (np.sum(np.sum(color_mask)), np.sum(np.sum(inv_mask)), np.sum(np.sum(color_mask))/np.sum(np.sum(inv_mask))),fontsize=10,ha='center')

            plt.title('hyper parameter', fontsize=10)
            axs[0].imshow(self.scaled)
            axs[0].set_title('original image')
            axs[0].axis("off")

            axs[1].imshow(self.scaled*color_mask)
            axs[1].set_title('apply a postive mask')
            axs[1].axis("off")
            
            axs[2].imshow(self.scaled*inv_mask)
            axs[2].set_title('apply a negative mask')
            axs[2].axis("off")
            #im = plt.imshow(self.scaled*color_mask,animated=True)
            #ims.append([im])
        else:
            print("wrong")
            #for gray
            plt.subplot(131)
            plt.imshow(self.scaled,cmap = 'gray')
            plt.subplot(132)
            plt.imshow(self.scaled*mask,cmap = 'gray')
            plt.axis('off')
            plt.subplot(133)
            plt.imshow(self.scaled*~mask,cmap = 'gray')
            plt.axis('off')
        return mask




if __name__ == '__main__':
    
    A = np.array([[1,2,3],[2,5,8],[3,8,9]])
    A = ImageSegmenter("actual_data_different_method_1.png")

    image = imread("checking_patch.png")
    mask = A.segment(r=50)
    plt.show()
    original_m,original_n = mask.shape
    """
    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    axs[0].imshow(mask, cmap="gray")
    mask = cv2.resize(np.float32(mask),(original_n*10,original_m*10))
    axs[1].imshow(mask)
    axs[2].imshow(image)
    """
    mask = cv2.resize(np.float32(mask),(original_n*10,original_m*10))
    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    mask = cv2.resize(np.float32(mask),(original_n*10,original_m*10))
    axs[0].imshow(image)
    axs[0].axis("off")
    axs[0].set_title("original tissue image")
    
    axs[1].imshow(mask)
    axs[1].axis("off")
    axs[1].set_title("Segmented Tissue")
    plt.suptitle("Tissue segmentation", fontsize=20)
    plt.show()
