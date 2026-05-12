# Spectral Graph Image Segmentation

A small Python demo for image segmentation using a graph-based spectral cut. The image is converted into a pixel graph, neighboring pixels are connected with weights based on brightness similarity and spatial distance, and the second eigenvector of the normalized graph Laplacian is used to split the image into two regions.

This was originally tested on a small tissue image sample.

## Example

Input image:

```text
actual_data_different_method_1.png
```

Example output:

```text
outputs/tissue_segmentation_result.png
outputs/tissue_segmentation_mask.png
```

## Environment

Tested with:

```text
Python 3.8+
Ubuntu 20.04
```

Install dependencies:

```bash
pip install numpy scipy matplotlib imageio opencv-python
```

## Run

Place the input image in the project root:

```text
actual_data_different_method_1.png
```

Then run:

```bash
python image_segmentation.py actual_data_different_method_1.png \
  --output outputs/tissue_segmentation_result.png \
  --mask-output outputs/tissue_segmentation_mask.png \
  --radius 6 \
  --sigma-b 0.03 \
  --sigma-x 5
```

## Reproduce the sample result

```bash
mkdir -p outputs

python image_segmentation.py actual_data_different_method_1.png \
  --output outputs/tissue_segmentation_result.png \
  --mask-output outputs/tissue_segmentation_mask.png \
  --radius 6 \
  --sigma-b 0.03 \
  --sigma-x 5
```

Expected behavior:

- The script builds a sparse adjacency matrix for the image graph.
- It computes the normalized graph Laplacian.
- It segments the image using the sign of the second eigenvector.
- It saves both a visualization and a binary mask.

## Parameters

| Parameter | Description | Example |
|---|---|---|
| `--radius` | Pixel neighborhood radius used to build graph edges | `6` |
| `--sigma-b` | Brightness similarity scale | `0.03` |
| `--sigma-x` | Spatial distance scale | `5` |
| `--output` | Path for the visualization result | `outputs/tissue_segmentation_result.png` |
| `--mask-output` | Path for the binary segmentation mask | `outputs/tissue_segmentation_mask.png` |

## Notes

This is a classical spectral graph segmentation demo, not a production biomedical segmentation model. It is useful for showing how graph-based methods can separate image regions without training data, but the result depends heavily on image size and hyperparameters.

For larger images, resize the image before running the script because spectral decomposition can become expensive.

## Repository Structure

```text
.
├── image_segmentation.py
├── actual_data_different_method_1.png
├── outputs/
│   ├── tissue_segmentation_result.png
│   └── tissue_segmentation_mask.png
└── README.md
```

## Acknowledgement

This project is based on a graph Laplacian / normalized cut approach for image segmentation.
