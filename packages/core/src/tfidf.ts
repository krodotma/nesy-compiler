import { tokenize } from './tokenizer';

export interface TFIDFResult {
  terms: string[];           // Vocabulary
  documents: string[];       // Document IDs
  matrix: number[][];        // TF-IDF matrix [doc][term]
  idf: Map<string, number>;  // IDF values
}

/**
 * Builds a TF-IDF matrix from a corpus of code files.
 *
 * @param corpus A map where keys are document IDs (e.g., file paths) and values are the code content.
 * @returns A TFIDFResult object containing the matrix and metadata.
 */
export function buildTFIDFMatrix(corpus: Map<string, string>): TFIDFResult {
  const documents = Array.from(corpus.keys());
  const N = documents.length;

  // 1. Tokenize all documents and compute raw term frequencies (TF) and document frequencies (DF).
  const docTermCounts = new Map<string, Map<string, number>>(); // docId -> (term -> count)
  const docTokenTotals = new Map<string, number>(); // docId -> total token count
  const globalDocFreq = new Map<string, number>(); // term -> number of docs containing term
  const vocabulary = new Set<string>();

  for (const [docId, content] of corpus) {
    const tokens = tokenize(content);
    const totalTokens = tokens.length;
    docTokenTotals.set(docId, totalTokens);

    const counts = new Map<string, number>();
    const seenInDoc = new Set<string>();

    for (const token of tokens) {
      vocabulary.add(token);
      counts.set(token, (counts.get(token) || 0) + 1);
      seenInDoc.add(token);
    }

    docTermCounts.set(docId, counts);

    // Update global document frequency
    for (const token of seenInDoc) {
      globalDocFreq.set(token, (globalDocFreq.get(token) || 0) + 1);
    }
  }

  // 2. Sort vocabulary for consistent matrix column ordering.
  const terms = Array.from(vocabulary).sort();

  // 3. Compute IDF for each term.
  // Using formula: IDF(t) = log(N / df(t))
  const idf = new Map<string, number>();
  for (const term of terms) {
    const df = globalDocFreq.get(term) || 0;
    // Standard IDF, assuming df >= 1 since terms come from corpus
    const idfVal = Math.log(N / (df || 1));
    idf.set(term, idfVal);
  }

  // 4. Build the TF-IDF Matrix [docIndex][termIndex]
  const matrix: number[][] = [];

  for (let i = 0; i < N; i++) {
    const docId = documents[i];
    const row: number[] = new Array(terms.length).fill(0);
    const counts = docTermCounts.get(docId);
    const totalTokens = docTokenTotals.get(docId) || 1;

    if (counts) {
      for (let j = 0; j < terms.length; j++) {
        const term = terms[j];
        const rawCount = counts.get(term) || 0;

        // TF: Normalized by document length
        const tf = rawCount / totalTokens;

        // IDF
        const termIdf = idf.get(term) || 0;

        row[j] = tf * termIdf;
      }
    }
    matrix.push(row);
  }

  return {
    terms,
    documents,
    matrix,
    idf
  };
}
