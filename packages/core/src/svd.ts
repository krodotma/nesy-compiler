/**
 * Lightweight SVD (Singular Value Decomposition) for LSA
 * Uses power iteration method for truncated SVD - suitable for term-document matrices.
 *
 * For a matrix A (m x n):
 *   A ≈ U * Σ * V^T
 * where:
 *   U (m x k): Left singular vectors (document vectors in LSA)
 *   Σ (k x k): Diagonal matrix of singular values
 *   V (n x k): Right singular vectors (term vectors in LSA)
 */

export interface SVDResult {
  U: number[][];      // Left singular vectors [m x k]
  S: number[];        // Singular values [k]
  V: number[][];      // Right singular vectors [n x k]
  k: number;          // Number of components
}

/**
 * Compute the dot product of two vectors.
 */
function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/**
 * Compute the L2 norm of a vector.
 */
function norm(v: number[]): number {
  return Math.sqrt(dot(v, v));
}

/**
 * Normalize a vector in place.
 */
function normalize(v: number[]): void {
  const n = norm(v);
  if (n > 1e-10) {
    for (let i = 0; i < v.length; i++) {
      v[i] /= n;
    }
  }
}

/**
 * Matrix-vector multiplication: A * v
 */
function matVec(A: number[][], v: number[]): number[] {
  const m = A.length;
  const result = new Array(m).fill(0);
  for (let i = 0; i < m; i++) {
    result[i] = dot(A[i], v);
  }
  return result;
}

/**
 * Transpose matrix-vector multiplication: A^T * v
 */
function matTVec(A: number[][], v: number[]): number[] {
  const n = A[0]?.length || 0;
  const result = new Array(n).fill(0);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < A.length; i++) {
      result[j] += A[i][j] * v[i];
    }
  }
  return result;
}

/**
 * Power iteration to find the dominant singular triplet.
 * Returns [u, sigma, v] where A*v ≈ sigma*u and A^T*u ≈ sigma*v
 */
function powerIteration(
  A: number[][],
  maxIter: number = 100,
  tolerance: number = 1e-8
): { u: number[]; sigma: number; v: number[] } {
  const m = A.length;
  const n = A[0]?.length || 0;

  // Initialize with random vector
  let v = new Array(n);
  for (let i = 0; i < n; i++) {
    v[i] = Math.random() - 0.5;
  }
  normalize(v);

  let sigma = 0;
  let u = new Array(m).fill(0);

  for (let iter = 0; iter < maxIter; iter++) {
    // u = A * v
    u = matVec(A, v);
    const newSigma = norm(u);

    if (newSigma < 1e-10) {
      // Matrix is effectively zero in this direction
      return { u: new Array(m).fill(0), sigma: 0, v };
    }

    normalize(u);

    // v = A^T * u
    const vNew = matTVec(A, u);
    normalize(vNew);

    // Check convergence
    if (Math.abs(newSigma - sigma) < tolerance) {
      return { u, sigma: newSigma, v: vNew };
    }

    sigma = newSigma;
    v = vNew;
  }

  return { u, sigma, v };
}

/**
 * Deflate matrix by removing contribution of a singular triplet.
 * A_new = A - sigma * u * v^T
 */
function deflate(A: number[][], u: number[], sigma: number, v: number[]): number[][] {
  const m = A.length;
  const n = A[0]?.length || 0;
  const result: number[][] = [];

  for (let i = 0; i < m; i++) {
    result[i] = new Array(n);
    for (let j = 0; j < n; j++) {
      result[i][j] = A[i][j] - sigma * u[i] * v[j];
    }
  }

  return result;
}

/**
 * Truncated SVD using sequential power iteration with deflation.
 *
 * @param matrix The input matrix (m x n)
 * @param k Number of singular values/vectors to compute (default: min(m, n, 100))
 * @param options Configuration options
 * @returns SVDResult with U, S, V matrices
 */
export function truncatedSVD(
  matrix: number[][],
  k?: number,
  options: { maxIter?: number; tolerance?: number } = {}
): SVDResult {
  const m = matrix.length;
  const n = matrix[0]?.length || 0;

  if (m === 0 || n === 0) {
    return { U: [], S: [], V: [], k: 0 };
  }

  const maxK = Math.min(m, n, k ?? 100);
  const { maxIter = 100, tolerance = 1e-8 } = options;

  const U: number[][] = [];
  const S: number[] = [];
  const V: number[][] = [];

  // Work on a copy to avoid mutating input
  let A = matrix.map(row => [...row]);

  for (let i = 0; i < maxK; i++) {
    const { u, sigma, v } = powerIteration(A, maxIter, tolerance);

    // Stop if singular value is effectively zero
    if (sigma < tolerance) {
      break;
    }

    U.push(u);
    S.push(sigma);
    V.push(v);

    // Deflate for next iteration
    A = deflate(A, u, sigma, v);
  }

  // Transpose U and V to match standard SVD output format
  // U should be [m x k], V should be [n x k]
  const UT: number[][] = [];
  const VT: number[][] = [];

  for (let i = 0; i < m; i++) {
    UT[i] = U.map(col => col[i]);
  }

  for (let j = 0; j < n; j++) {
    VT[j] = V.map(col => col[j]);
  }

  return {
    U: UT,
    S,
    V: VT,
    k: S.length
  };
}

/**
 * Reconstruct the original matrix from SVD components (for verification).
 * A ≈ U * diag(S) * V^T
 */
export function reconstructFromSVD(svd: SVDResult): number[][] {
  const { U, S, V, k } = svd;
  const m = U.length;
  const n = V.length;

  const result: number[][] = [];
  for (let i = 0; i < m; i++) {
    result[i] = new Array(n).fill(0);
    for (let j = 0; j < n; j++) {
      for (let l = 0; l < k; l++) {
        result[i][j] += U[i][l] * S[l] * V[j][l];
      }
    }
  }

  return result;
}

/**
 * Compute the Frobenius norm error between original and reconstructed matrix.
 */
export function reconstructionError(original: number[][], svd: SVDResult): number {
  const reconstructed = reconstructFromSVD(svd);
  let sumSq = 0;

  for (let i = 0; i < original.length; i++) {
    for (let j = 0; j < original[i].length; j++) {
      const diff = original[i][j] - (reconstructed[i]?.[j] ?? 0);
      sumSq += diff * diff;
    }
  }

  return Math.sqrt(sumSq);
}
