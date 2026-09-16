# Geometric Shape Detector

A lightweight, zero-dependency computer vision engine built entirely from scratch in TypeScript and HTML5 Canvas. Detects, segments, localizes, and classifies 2D geometric shapes (circles, triangles, rectangles, pentagons, and stars) in real time with sub-20ms processing times.

---

## Features

- **Zero External CV Libraries**: Pure TypeScript math and pixel-level operations (no OpenCV, TensorFlow, or Python dependencies).
- **5-Stage Computer Vision Pipeline**:
  1. **Adaptive Binarization**: Auto-samples border pixels to determine background polarity (supports light-on-dark, dark-on-light, and transparency).
  2. **Connected Component Labeling (CCL)**: Fast BFS-based flood fill with adaptive noise filtering.
  3. **True Moore-Neighbor Contour Tracing**: Robust boundary traversal with background backtracking and Jacob's stopping criterion (accurately traces sharp corners, rotated polygons, and outer perimeters).
  4. **Ramer-Douglas-Peucker (RDP) Simplification**: Multi-scale polygonal approximation.
  5. **Geometric Classification**: Classifies shapes based on vertex topology, circularity index ($4\pi A / P^2$), and alternating radial star ratios.
- **Rich Canvas Overlays**: Real-time bounding boxes, centroid coordinates, and confidence badges drawn directly over images.
- **Built-in Evaluation Suite**: Interactive benchmark tool measuring precision, recall, F1-score, and IoU against ground truth data.
- **Custom Image Upload**: Test custom diagrams, flowcharts, circuit schematics, or photos.

---

## Shapes Detected

| Shape | Classification Strategy |
| :--- | :--- |
| **Circle** | High circularity index ($> 0.78$) with smooth curvature |
| **Triangle** | Exact 3-vertex RDP simplification |
| **Rectangle / Square** | Exact 4-vertex RDP simplification (rotation-invariant) |
| **Pentagon** | Exact 5-vertex RDP simplification |
| **Star** | 10-vertex simplification with alternating radial distance ratio |

---

## Evaluation Benchmark

The detector achieves **100% detection and classification accuracy** against the included ground truth dataset:

| Test Scenario | Expected Shapes | Detected | Status |
| :--- | :--- | :--- | :--- |
| `circle_simple.png` | 1 Circle | Circle (97% conf) | Passed |
| `triangle_basic.png` | 1 Triangle | Triangle (92% conf) | Passed |
| `rectangle_square.png` | 1 Rectangle | Rectangle (95% conf) | Passed |
| `pentagon_regular.png` | 1 Pentagon | Pentagon (90% conf) | Passed |
| `star_five_point.png` | 1 Star | Star (95% conf) | Passed |
| `mixed_shapes_simple.png` | Triangle, Circle, Rectangle | All 3 detected | Passed |
| `complex_scene.png` | Circle, Rectangle, Star | All 3 detected | Passed |
| `edge_cases.png` | Small Triangle, Rotated Rectangle | Both detected | Passed |
| `noisy_background.png` | Pentagon, Circle | Both detected | Passed |
| `no_shapes.png` | 0 Shapes | 0 (no false positives) | Passed |

---

## Getting Started

### Prerequisites
- Node.js (version 18 or higher)
- npm or yarn

### Installation & Run

```bash
# 1. Install dependencies
npm install

# 2. Start Vite development server
npm run dev

# 3. Build for production
npm run build
```

Open [http://localhost:5174](http://localhost:5174) in your browser.

---

## Project Structure

```
Shape-detector-personal/
├── public/
│   ├── ground_truth.json     # Ground truth benchmark annotations
│   └── vite.svg
├── src/
│   ├── main.ts               # Core CV algorithm & main UI controller
│   ├── evaluation.ts         # Benchmark evaluation runner
│   ├── evaluation-manager.ts # Modal and evaluation trigger manager
│   ├── evaluation-utils.ts   # IoU, distance, and precision/recall metrics
│   ├── test-images-data.ts   # Vector test image dataset
│   ├── ui-utils.ts           # Selection controls and UI helpers
│   └── style.css             # Application styling
├── upload/                   # Sample real-world test images
├── ground_truth.json         # Benchmark dataset reference
├── index.html                # App layout and canvas viewports
├── tsconfig.json             # TypeScript configuration
└── package.json              # Scripts and dev dependencies
```
