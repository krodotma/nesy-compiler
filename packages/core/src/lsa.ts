/**
 * Latent Semantic Analysis (LSA) for code understanding.
 *
 * LSA Pipeline:
 * 1. Tokenize code files → tokens
 * 2. Build TF-IDF matrix → term-document matrix
 * 3. Apply SVD → reduce dimensionality
 * 4. Project documents/terms into semantic space
 *
 * Use cases:
 * - Semantic similarity between code files
 * - Concept discovery across codebase
 * - Thrash detection (semantically redundant code)
 */

import { buildTFIDFMatrix, TFIDFResult } from './tfidf';
import { truncatedSVD, SVDResult } from './svd';

export interface LSAModel {
  /** Original TF-IDF data */
  tfidf: TFIDFResult;
  /** SVD decomposition */
  svd: SVDResult;
  /** Number of latent dimensions */
  dimensions: number;
  /** Document vectors in latent space [doc x k] */
  documentVectors: number[][];
  /** Term vectors in latent space [term x k] */
  termVectors: number[][];
}

export interface LSAConfig {
  /** Number of latent dimensions (default: 100 or min(docs, terms)) */
  dimensions?: number;
  /** SVD convergence tolerance */
  tolerance?: number;
  /** Maximum SVD iterations */
  maxIterations?: number;
}

const DEFAULT_LSA_CONFIG: LSAConfig = {
  dimensions: 100,
  tolerance: 1e-8,
  maxIterations: 100
};

/**
 * Build an LSA model from a corpus of code files.
 *
 * @param corpus Map of document IDs to code content
 * @param config LSA configuration options
 * @returns LSAModel with document and term vectors in latent space
 */
export function buildLSAModel(
  corpus: Map<string, string>,
  config?: Partial<LSAConfig>
): LSAModel {
  const cfg = { ...DEFAULT_LSA_CONFIG, ...config };

  // Step 1 & 2: Build TF-IDF matrix
  const tfidf = buildTFIDFMatrix(corpus);

  if (tfidf.documents.length === 0 || tfidf.terms.length === 0) {
    return {
      tfidf,
      svd: { U: [], S: [], V: [], k: 0 },
      dimensions: 0,
      documentVectors: [],
      termVectors: []
    };
  }

  // Determine actual dimensions
  const maxDims = Math.min(
    tfidf.documents.length,
    tfidf.terms.length,
    cfg.dimensions || 100
  );

  // Step 3: Apply truncated SVD
  const svd = truncatedSVD(tfidf.matrix, maxDims, {
    maxIter: cfg.maxIterations,
    tolerance: cfg.tolerance
  });

  // Step 4: Compute document and term vectors
  // Document vectors: U * sqrt(S)
  // Term vectors: V * sqrt(S)
  const sqrtS = svd.S.map(Math.sqrt);

  const documentVectors = svd.U.map(row =>
    row.map((val, i) => val * sqrtS[i])
  );

  const termVectors = svd.V.map(row =>
    row.map((val, i) => val * sqrtS[i])
  );

  return {
    tfidf,
    svd,
    dimensions: svd.k,
    documentVectors,
    termVectors
  };
}

/**
 * Compute cosine similarity between two vectors.
 */
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator > 0 ? dotProduct / denominator : 0;
}

/**
 * Get the semantic similarity between two documents.
 *
 * @param model The LSA model
 * @param docId1 First document ID
 * @param docId2 Second document ID
 * @returns Cosine similarity in latent space (0-1)
 */
export function documentSimilarity(
  model: LSAModel,
  docId1: string,
  docId2: string
): number {
  const idx1 = model.tfidf.documents.indexOf(docId1);
  const idx2 = model.tfidf.documents.indexOf(docId2);

  if (idx1 < 0 || idx2 < 0) return 0;

  return cosineSimilarity(
    model.documentVectors[idx1],
    model.documentVectors[idx2]
  );
}

/**
 * Get the semantic similarity between two terms.
 *
 * @param model The LSA model
 * @param term1 First term
 * @param term2 Second term
 * @returns Cosine similarity in latent space (0-1)
 */
export function termSimilarity(
  model: LSAModel,
  term1: string,
  term2: string
): number {
  const idx1 = model.tfidf.terms.indexOf(term1);
  const idx2 = model.tfidf.terms.indexOf(term2);

  if (idx1 < 0 || idx2 < 0) return 0;

  return cosineSimilarity(
    model.termVectors[idx1],
    model.termVectors[idx2]
  );
}

/**
 * Find the most similar documents to a given document.
 *
 * @param model The LSA model
 * @param docId Document ID to query
 * @param topK Number of results to return
 * @returns Array of [docId, similarity] sorted by similarity descending
 */
export function findSimilarDocuments(
  model: LSAModel,
  docId: string,
  topK: number = 10
): Array<[string, number]> {
  const idx = model.tfidf.documents.indexOf(docId);
  if (idx < 0) return [];

  const queryVector = model.documentVectors[idx];
  const similarities: Array<[string, number]> = [];

  for (let i = 0; i < model.tfidf.documents.length; i++) {
    if (i === idx) continue;
    const sim = cosineSimilarity(queryVector, model.documentVectors[i]);
    similarities.push([model.tfidf.documents[i], sim]);
  }

  return similarities
    .sort((a, b) => b[1] - a[1])
    .slice(0, topK);
}

/**
 * Find terms most related to a given term.
 *
 * @param model The LSA model
 * @param term Term to query
 * @param topK Number of results to return
 * @returns Array of [term, similarity] sorted by similarity descending
 */
export function findRelatedTerms(
  model: LSAModel,
  term: string,
  topK: number = 10
): Array<[string, number]> {
  const idx = model.tfidf.terms.indexOf(term);
  if (idx < 0) return [];

  const queryVector = model.termVectors[idx];
  const similarities: Array<[string, number]> = [];

  for (let i = 0; i < model.tfidf.terms.length; i++) {
    if (i === idx) continue;
    const sim = cosineSimilarity(queryVector, model.termVectors[i]);
    similarities.push([model.tfidf.terms[i], sim]);
  }

  return similarities
    .sort((a, b) => b[1] - a[1])
    .slice(0, topK);
}

/**
 * Project a new document into the LSA space.
 * Useful for querying documents not in the original corpus.
 *
 * @param model The LSA model
 * @param tokens Pre-tokenized document
 * @returns Document vector in latent space
 */
export function projectDocument(
  model: LSAModel,
  tokens: string[]
): number[] {
  if (model.dimensions === 0) return [];

  // Build term frequency for the new document
  const termCounts = new Map<string, number>();
  for (const token of tokens) {
    termCounts.set(token, (termCounts.get(token) || 0) + 1);
  }

  const totalTokens = tokens.length || 1;

  // Build TF-IDF vector for the new document
  const tfidfVector = new Array(model.tfidf.terms.length).fill(0);
  for (let i = 0; i < model.tfidf.terms.length; i++) {
    const term = model.tfidf.terms[i];
    const count = termCounts.get(term) || 0;
    const tf = count / totalTokens;
    const idf = model.tfidf.idf.get(term) || 0;
    tfidfVector[i] = tf * idf;
  }

  // Project: q_lsa = q_tfidf * V * S^(-1)
  const result = new Array(model.dimensions).fill(0);
  for (let k = 0; k < model.dimensions; k++) {
    const sInv = model.svd.S[k] > 1e-10 ? 1 / model.svd.S[k] : 0;
    for (let j = 0; j < model.tfidf.terms.length; j++) {
      result[k] += tfidfVector[j] * model.termVectors[j][k] * sInv;
    }
    // Scale by sqrt(S) to match document vectors
    result[k] *= Math.sqrt(model.svd.S[k]);
  }

  return result;
}

/**
 * Detect semantically redundant documents (potential "thrash").
 * Documents with high similarity to others may represent redundant code.
 *
 * @param model The LSA model
 * @param threshold Similarity threshold (default: 0.9)
 * @returns Array of document pairs with similarity above threshold
 */
export function detectRedundancy(
  model: LSAModel,
  threshold: number = 0.9
): Array<{ doc1: string; doc2: string; similarity: number }> {
  const redundant: Array<{ doc1: string; doc2: string; similarity: number }> = [];

  for (let i = 0; i < model.tfidf.documents.length; i++) {
    for (let j = i + 1; j < model.tfidf.documents.length; j++) {
      const sim = cosineSimilarity(
        model.documentVectors[i],
        model.documentVectors[j]
      );
      if (sim >= threshold) {
        redundant.push({
          doc1: model.tfidf.documents[i],
          doc2: model.tfidf.documents[j],
          similarity: sim
        });
      }
    }
  }

  return redundant.sort((a, b) => b.similarity - a.similarity);
}

/**
 * Get the top concepts (latent dimensions) for a document.
 * Each concept represents a semantic theme in the codebase.
 *
 * @param model The LSA model
 * @param docId Document ID
 * @param topK Number of concepts to return
 * @returns Array of [conceptIndex, weight] sorted by absolute weight
 */
export function getDocumentConcepts(
  model: LSAModel,
  docId: string,
  topK: number = 5
): Array<[number, number]> {
  const idx = model.tfidf.documents.indexOf(docId);
  if (idx < 0) return [];

  const vector = model.documentVectors[idx];
  const indexed: Array<[number, number]> = vector.map((val, i) => [i, val]);

  return indexed
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, topK);
}
