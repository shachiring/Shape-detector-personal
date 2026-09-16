import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";

export interface Point {
  x: number;
  y: number;
}

export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star";
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: Point;
  area: number;
}

export interface DetectionResult {
  shapes: DetectedShape[];
  processingTime: number;
  imageWidth: number;
  imageHeight: number;
}

export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  // ───────────────────────────────────────────────────────────
  //  STAGE 1: Binarization with auto-polarity background detection
  // ───────────────────────────────────────────────────────────
  private binarize(
    imageData: ImageData,
    threshold: number = 128
  ): boolean[][] {
    const { width, height, data } = imageData;
    const grid: boolean[][] = Array.from({ length: height }, () =>
      new Array<boolean>(width).fill(false)
    );

    // Sample border pixels to detect background polarity (dark vs light bg)
    let borderSum = 0;
    let borderCount = 0;
    for (let x = 0; x < width; x += 2) {
      const topIdx = x * 4;
      const btmIdx = ((height - 1) * width + x) * 4;
      borderSum += 0.299 * data[topIdx] + 0.587 * data[topIdx + 1] + 0.114 * data[topIdx + 2];
      borderSum += 0.299 * data[btmIdx] + 0.587 * data[btmIdx + 1] + 0.114 * data[btmIdx + 2];
      borderCount += 2;
    }
    for (let y = 0; y < height; y += 2) {
      const lftIdx = (y * width) * 4;
      const rgtIdx = (y * width + (width - 1)) * 4;
      borderSum += 0.299 * data[lftIdx] + 0.587 * data[lftIdx + 1] + 0.114 * data[lftIdx + 2];
      borderSum += 0.299 * data[rgtIdx] + 0.587 * data[rgtIdx + 1] + 0.114 * data[rgtIdx + 2];
      borderCount += 2;
    }
    const avgBg = borderSum / (borderCount || 1);
    const isDarkBg = avgBg < 90;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 4;
        const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        const alpha = data[i + 3];

        if (alpha < 64) {
          grid[y][x] = false;
        } else if (isDarkBg) {
          // Foreground is brighter than dark background
          grid[y][x] = gray > Math.max(100, avgBg + 40);
        } else {
          // Foreground is darker than light background
          grid[y][x] = gray < threshold;
        }
      }
    }
    return grid;
  }

  // ───────────────────────────────────────────────────────────
  //  STAGE 2: Connected-component labeling via BFS flood-fill
  // ───────────────────────────────────────────────────────────
  private findBlobs(
    grid: boolean[][],
    minArea: number = 50
  ): Point[][] {
    const height = grid.length;
    const width = grid[0].length;
    const visited: boolean[][] = Array.from({ length: height }, () =>
      new Array<boolean>(width).fill(false)
    );
    const blobs: Point[][] = [];

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (grid[y][x] && !visited[y][x]) {
          const blob: Point[] = [];
          const queue: Point[] = [{ x, y }];
          visited[y][x] = true;

          while (queue.length > 0) {
            const p = queue.pop()!;
            blob.push(p);

            const neighbors: Point[] = [
              { x: p.x - 1, y: p.y },
              { x: p.x + 1, y: p.y },
              { x: p.x, y: p.y - 1 },
              { x: p.x, y: p.y + 1 },
            ];
            for (const n of neighbors) {
              if (
                n.x >= 0 && n.x < width &&
                n.y >= 0 && n.y < height &&
                grid[n.y][n.x] && !visited[n.y][n.x]
              ) {
                visited[n.y][n.x] = true;
                queue.push(n);
              }
            }
          }

          if (blob.length >= minArea) {
            blobs.push(blob);
          }
        }
      }
    }
    return blobs;
  }

  // ───────────────────────────────────────────────────────────
  //  STAGE 3: True Moore-Neighbor Contour Tracing
  // ───────────────────────────────────────────────────────────
  private traceContour(grid: boolean[][], blob: Point[]): Point[] {
    const height = grid.length;
    const width = grid[0].length;
    let startX = Infinity, startY = Infinity;
    for (const p of blob) {
      if (p.y < startY || (p.y === startY && p.x < startX)) {
        startX = p.x;
        startY = p.y;
      }
    }

    // 8 clockwise directions starting from North (0, -1)
    const cdx = [0, 1, 1, 1, 0, -1, -1, -1];
    const cdy = [-1, -1, 0, 1, 1, 1, 0, -1];

    const contour: Point[] = [];
    let currX = startX;
    let currY = startY;
    let backDir = 0; // Known background direction
    let firstStep: { fromX: number; fromY: number; toX: number; toY: number } | null = null;
    const maxIter = blob.length * 4;
    let iter = 0;

    while (iter < maxIter) {
      contour.push({ x: currX, y: currY });

      let nextX = -1, nextY = -1;
      let nextBackDir = -1;
      let found = false;

      for (let i = 0; i < 8; i++) {
        const d = (backDir + i) % 8;
        const nx = currX + cdx[d];
        const ny = currY + cdy[d];

        if (nx >= 0 && nx < width && ny >= 0 && ny < height && grid[ny][nx]) {
          nextX = nx;
          nextY = ny;
          const lastBgDir = (d + 7) % 8;
          const bgX = currX + cdx[lastBgDir];
          const bgY = currY + cdy[lastBgDir];
          const rdx = bgX - nextX;
          const rdy = bgY - nextY;
          for (let k = 0; k < 8; k++) {
            if (cdx[k] === rdx && cdy[k] === rdy) {
              nextBackDir = k;
              break;
            }
          }
          found = true;
          break;
        }
      }

      if (!found) break;

      if (!firstStep) {
        firstStep = { fromX: currX, fromY: currY, toX: nextX, toY: nextY };
      } else if (
        currX === firstStep.fromX &&
        currY === firstStep.fromY &&
        nextX === firstStep.toX &&
        nextY === firstStep.toY
      ) {
        // Jacob's stopping criterion
        break;
      }

      currX = nextX;
      currY = nextY;
      backDir = nextBackDir !== -1 ? nextBackDir : 0;
      iter++;
    }

    return contour;
  }

  // ───────────────────────────────────────────────────────────
  //  STAGE 4: Ramer-Douglas-Peucker simplification
  // ───────────────────────────────────────────────────────────
  private rdpSimplify(points: Point[], epsilon: number): Point[] {
    if (points.length <= 2) return points.slice();

    let maxDist = 0;
    let maxIdx = 0;
    const first = points[0];
    const last = points[points.length - 1];

    for (let i = 1; i < points.length - 1; i++) {
      const d = this.pointToLineDistance(points[i], first, last);
      if (d > maxDist) {
        maxDist = d;
        maxIdx = i;
      }
    }

    if (maxDist > epsilon) {
      const left = this.rdpSimplify(points.slice(0, maxIdx + 1), epsilon);
      const right = this.rdpSimplify(points.slice(maxIdx), epsilon);
      return left.slice(0, -1).concat(right);
    } else {
      return [first, last];
    }
  }

  private pointToLineDistance(p: Point, a: Point, b: Point): number {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const lenSq = dx * dx + dy * dy;
    if (lenSq === 0) {
      return Math.sqrt((p.x - a.x) ** 2 + (p.y - a.y) ** 2);
    }
    const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / lenSq));
    const projX = a.x + t * dx;
    const projY = a.y + t * dy;
    return Math.sqrt((p.x - projX) ** 2 + (p.y - projY) ** 2);
  }

  private polygonArea(vertices: Point[]): number {
    let area = 0;
    const n = vertices.length;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      area += vertices[i].x * vertices[j].y;
      area -= vertices[j].x * vertices[i].y;
    }
    return Math.abs(area) / 2;
  }

  private polygonPerimeter(vertices: Point[]): number {
    let perimeter = 0;
    const n = vertices.length;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      perimeter += Math.sqrt(
        (vertices[j].x - vertices[i].x) ** 2 +
        (vertices[j].y - vertices[i].y) ** 2
      );
    }
    return perimeter;
  }

  private centroid(points: Point[]): Point {
    let sx = 0, sy = 0;
    for (const p of points) {
      sx += p.x;
      sy += p.y;
    }
    return { x: sx / points.length, y: sy / points.length };
  }

  private boundingBox(points: Point[]): {
    x: number;
    y: number;
    width: number;
    height: number;
  } {
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    for (const p of points) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
  }

  private distancesToCentroid(vertices: Point[], center: Point): number[] {
    return vertices.map(
      (v) => Math.sqrt((v.x - center.x) ** 2 + (v.y - center.y) ** 2)
    );
  }

  // ───────────────────────────────────────────────────────────
  //  STAGE 5: Shape Classification
  // ───────────────────────────────────────────────────────────
  private classifyShape(
    contour: Point[],
    _blobPixelCount: number
  ): { type: DetectedShape["type"]; confidence: number } | null {
    const perimeter = this.polygonPerimeter(contour);
    const area = this.polygonArea(contour);

    // Circularity: 4π·A / P² (1.0 for circle, 0.785 for square)
    const circularity = (4 * Math.PI * area) / (perimeter * perimeter);

    const bb = this.boundingBox(contour);
    const diagLen = Math.sqrt(bb.width ** 2 + bb.height ** 2);

    const epsilons = [
      diagLen * 0.015,
      diagLen * 0.02,
      diagLen * 0.025,
      diagLen * 0.03,
      diagLen * 0.04,
      diagLen * 0.05,
    ];

    const simplifications = epsilons.map((eps) => {
      const simplified = this.rdpSimplify(contour, eps);
      let pts = simplified;
      if (pts.length > 1) {
        const first = pts[0];
        const last = pts[pts.length - 1];
        const d = Math.sqrt((first.x - last.x) ** 2 + (first.y - last.y) ** 2);
        if (d < eps * 2) {
          pts = pts.slice(0, -1);
        }
      }
      return { epsilon: eps, vertices: pts, count: pts.length };
    });

    const counts = simplifications.map((s) => s.count);

    // 1. Star Detection: ~10 vertices with alternating radial distance pattern
    for (const s of simplifications) {
      if (s.count >= 8 && s.count <= 12) {
        const center = this.centroid(s.vertices);
        const dists = this.distancesToCentroid(s.vertices, center);
        let alternating = 0;
        for (let i = 0; i < dists.length; i++) {
          const prev = dists[(i - 1 + dists.length) % dists.length];
          const curr = dists[i];
          const next = dists[(i + 1) % dists.length];
          if ((curr > prev && curr > next) || (curr < prev && curr < next)) {
            alternating++;
          }
        }
        const alternatingRatio = alternating / dists.length;
        if (alternatingRatio >= 0.6) {
          return {
            type: "star",
            confidence: Math.min(0.95, 0.7 + alternatingRatio * 0.25),
          };
        }
      }
    }

    // 2. Triangle Detection: exactly 3 vertices in simplification
    if (counts.includes(3)) {
      return { type: "triangle", confidence: 0.92 };
    }

    // 3. Rectangle Detection: 4 vertices in simplification
    if (counts.includes(4)) {
      return { type: "rectangle", confidence: 0.95 };
    }

    // 4. Pentagon Detection: 5 vertices in simplification
    if (counts.includes(5)) {
      return { type: "pentagon", confidence: 0.90 };
    }

    // 5. Circle Detection: high circularity and smooth curve
    if (circularity > 0.78) {
      return {
        type: "circle",
        confidence: Math.min(0.98, 0.7 + circularity * 0.3),
      };
    }

    // Fallbacks based on coarsest simplification
    const coarsest = simplifications[simplifications.length - 1];
    if (coarsest.count === 3) return { type: "triangle", confidence: 0.75 };
    if (coarsest.count === 4) return { type: "rectangle", confidence: 0.75 };
    if (coarsest.count === 5) return { type: "pentagon", confidence: 0.75 };
    if (circularity > 0.65) return { type: "circle", confidence: 0.70 };

    return null;
  }

  // ═══════════════════════════════════════════════════════════
  //  MAIN DETECTION ALGORITHM
  // ═══════════════════════════════════════════════════════════
  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();
    const shapes: DetectedShape[] = [];

    // Stage 1: Binarize
    const grid = this.binarize(imageData, 128);

    // Stage 2: Connected Blobs (adaptive minArea)
    const minArea = Math.max(30, Math.floor(imageData.width * imageData.height * 0.0001));
    const blobs = this.findBlobs(grid, minArea);

    for (const blob of blobs) {
      // Stage 3: Trace contour
      const contour = this.traceContour(grid, blob);
      if (contour.length < 10) continue;

      // Stage 4 & 5: Classify
      const classification = this.classifyShape(contour, blob.length);
      if (!classification) continue;

      const bb = this.boundingBox(contour);
      let center: Point;
      let area: number;

      if (classification.type === "circle") {
        const radius = (bb.width + bb.height) / 4;
        center = { x: bb.x + bb.width / 2, y: bb.y + bb.height / 2 };
        area = Math.PI * radius * radius;
      } else if (classification.type === "rectangle") {
        center = { x: bb.x + bb.width / 2, y: bb.y + bb.height / 2 };
        area = bb.width * bb.height;
      } else {
        center = this.centroid(contour);
        area = this.polygonArea(contour);
        if (Math.abs(area - blob.length) / Math.max(area, blob.length) > 0.35) {
          area = blob.length;
        }
      }

      shapes.push({
        type: classification.type,
        confidence: classification.confidence,
        boundingBox: bb,
        center,
        area,
      });
    }

    const processingTime = performance.now() - startTime;

    return {
      shapes,
      processingTime,
      imageWidth: imageData.width,
      imageHeight: imageData.height,
    };
  }

  // ───────────────────────────────────────────────────────────
  //  Draw detected shapes directly on canvas for rich visual feedback
  // ───────────────────────────────────────────────────────────
  drawDetections(results: DetectionResult): void {
    const { shapes } = results;
    if (shapes.length === 0) return;

    this.ctx.save();

    shapes.forEach((shape) => {
      const { x, y, width, height } = shape.boundingBox;

      // Draw bounding box
      this.ctx.strokeStyle = "#10b981"; // Emerald green
      this.ctx.lineWidth = 2;
      this.ctx.strokeRect(x, y, width, height);

      // Draw centroid dot
      this.ctx.fillStyle = "#ef4444"; // Red
      this.ctx.beginPath();
      this.ctx.arc(shape.center.x, shape.center.y, 4, 0, 2 * Math.PI);
      this.ctx.fill();

      // Draw badge label
      const label = `${shape.type.toUpperCase()} ${(shape.confidence * 100).toFixed(0)}%`;
      this.ctx.font = "bold 11px system-ui, sans-serif";
      const textWidth = this.ctx.measureText(label).width;
      const labelX = Math.max(0, x);
      const labelY = Math.max(16, y - 4);

      this.ctx.fillStyle = "#10b981";
      this.ctx.fillRect(labelX, labelY - 14, textWidth + 8, 16);

      this.ctx.fillStyle = "#ffffff";
      this.ctx.fillText(label, labelX + 4, labelY - 2);
    });

    this.ctx.restore();
  }

  loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      const objectUrl = URL.createObjectURL(file);

      img.onload = () => {
        const width = img.naturalWidth || img.width || 200;
        const height = img.naturalHeight || img.height || 200;

        this.canvas.width = width;
        this.canvas.height = height;
        this.ctx.drawImage(img, 0, 0, width, height);
        URL.revokeObjectURL(objectUrl);

        const imageData = this.ctx.getImageData(0, 0, width, height);
        resolve(imageData);
      };

      img.onerror = (err) => {
        URL.revokeObjectURL(objectUrl);
        reject(err);
      };

      img.src = objectUrl;
    });
  }
}

class ShapeDetectionApp {
  private detector: ShapeDetector;
  private imageInput: HTMLInputElement;
  private resultsDiv: HTMLDivElement;
  private testImagesDiv: HTMLDivElement;
  private evaluateButton: HTMLButtonElement;
  private evaluationResultsDiv: HTMLDivElement;
  private selectionManager: SelectionManager;
  private evaluationManager: EvaluationManager;

  constructor() {
    const canvas = document.getElementById(
      "originalCanvas"
    ) as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);

    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.resultsDiv = document.getElementById("results") as HTMLDivElement;
    this.testImagesDiv = document.getElementById(
      "testImages"
    ) as HTMLDivElement;
    this.evaluateButton = document.getElementById(
      "evaluateButton"
    ) as HTMLButtonElement;
    this.evaluationResultsDiv = document.getElementById(
      "evaluationResults"
    ) as HTMLDivElement;

    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(
      this.detector,
      this.evaluateButton,
      this.evaluationResultsDiv
    );

    this.setupEventListeners();
    this.loadTestImages().catch(console.error);
  }

  private setupEventListeners(): void {
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) {
        await this.processImage(file);
      }
    });

    this.evaluateButton.addEventListener("click", async () => {
      const selectedImages = this.selectionManager.getSelectedImages();
      await this.evaluationManager.runSelectedEvaluation(selectedImages);
    });
  }

  private async processImage(file: File): Promise<void> {
    try {
      this.resultsDiv.innerHTML = "<p>Processing...</p>";

      const imageData = await this.detector.loadImage(file);
      const results = await this.detector.detectShapes(imageData);

      this.displayResults(results);
      this.detector.drawDetections(results);
    } catch (error) {
      this.resultsDiv.innerHTML = `<p>Error: ${error}</p>`;
    }
  }

  private displayResults(results: DetectionResult): void {
    const { shapes, processingTime } = results;

    let html = `
      <p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}ms</p>
      <p><strong>Shapes Found:</strong> ${shapes.length}</p>
    `;

    if (shapes.length > 0) {
      html += "<h4>Detected Shapes:</h4><ul>";
      shapes.forEach((shape) => {
        html += `
          <li>
            <strong>${
              shape.type.charAt(0).toUpperCase() + shape.type.slice(1)
            }</strong><br>
            Confidence: ${(shape.confidence * 100).toFixed(1)}%<br>
            Center: (${shape.center.x.toFixed(1)}, ${shape.center.y.toFixed(1)})<br>
            Area: ${shape.area.toFixed(1)}px²<br>
            Bounding Box: [x: ${shape.boundingBox.x.toFixed(0)}, y: ${shape.boundingBox.y.toFixed(0)}, w: ${shape.boundingBox.width.toFixed(0)}, h: ${shape.boundingBox.height.toFixed(0)}]
          </li>
        `;
      });
      html += "</ul>";
    } else {
      html += "<p>No geometric shapes detected in this image.</p>";
    }

    this.resultsDiv.innerHTML = html;
  }

  private async loadTestImages(): Promise<void> {
    try {
      const module = await import("./test-images-data.js");
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();

      let html =
        '<h4>Click to upload your own image or use test images for detection. Right-click test images to select/deselect for evaluation:</h4><div class="evaluation-controls"><button id="selectAllBtn">Select All</button><button id="deselectAllBtn">Deselect All</button><span class="selection-info">0 images selected</span></div><div class="test-images-grid">';

      // Add upload functionality as first grid item
      html += `
        <div class="test-image-item upload-item" onclick="triggerFileUpload()">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Upload Image</div>
          <div class="upload-subtext">Click to select file</div>
        </div>
      `;

      imageNames.forEach((imageName) => {
        const dataUrl = testImages[imageName as keyof typeof testImages];
        const displayName = imageName
          .replace(/[_-]/g, " ")
          .replace(/\.(svg|png)$/i, "");
        html += `
          <div class="test-image-item" data-image="${imageName}" 
               onclick="loadTestImage('${imageName}', '${dataUrl}')" 
               oncontextmenu="toggleImageSelection(event, '${imageName}')">
            <img src="${dataUrl}" alt="${imageName}">
            <div>${displayName}</div>
          </div>
        `;
      });

      html += "</div>";
      this.testImagesDiv.innerHTML = html;

      this.selectionManager.setupSelectionControls();

      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        try {
          const response = await fetch(dataUrl);
          const blob = await response.blob();
          const file = new File([blob], name, { type: "image/svg+xml" });

          const imageData = await this.detector.loadImage(file);
          const results = await this.detector.detectShapes(imageData);
          this.displayResults(results);
          this.detector.drawDetections(results);

          console.log(`Loaded test image: ${name}`);
        } catch (error) {
          console.error("Error loading test image:", error);
        }
      };

      (window as any).toggleImageSelection = (
        event: MouseEvent,
        imageName: string
      ) => {
        event.preventDefault();
        this.selectionManager.toggleImageSelection(imageName);
      };

      (window as any).triggerFileUpload = () => {
        this.imageInput.click();
      };
    } catch (error) {
      this.testImagesDiv.innerHTML = `
        <p>Test images not available. Run 'node convert-svg-to-png.js' to generate test image data.</p>
        <p>SVG files are available in the test-images/ directory.</p>
      `;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ShapeDetectionApp();
});
