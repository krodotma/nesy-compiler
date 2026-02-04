import { tokenize } from "../tokenizer";
import { buildTFIDFMatrix } from "../tfidf";
import { describe, it, expect } from "vitest";

describe("Tokenizer", () => {
  it("should split CamelCase and snake_case", () => {
    const code = "class MySuperClass { const my_variable = 1; }";
    const tokens = tokenize(code);
    // Based on implementation: 'class', 'my', 'super', 'class', 'const', 'my', 'variable'
    expect(tokens).toContain("super");
    expect(tokens).toContain("class");
    expect(tokens).toContain("variable");
    expect(tokens).toContain("my");
  });

  it("should handle mixed cases", () => {
    const code = "XMLHttpRequest_handler";
    const tokens = tokenize(code);
    expect(tokens).toContain("xml");
    expect(tokens).toContain("http");
    expect(tokens).toContain("request");
    expect(tokens).toContain("handler");
  });
});

describe("TF-IDF", () => {
  it("should calculate scores correctly", () => {
    const corpus = new Map<string, string>();
    corpus.set("doc1", "apple banana apple");
    corpus.set("doc2", "banana cherry");

    const result = buildTFIDFMatrix(corpus);
    
    const appleIdx = result.terms.indexOf("apple");
    const bananaIdx = result.terms.indexOf("banana");
    
    expect(appleIdx).toBeGreaterThan(-1);
    expect(bananaIdx).toBeGreaterThan(-1);

    const doc1Row = result.matrix[0]; // doc1
    const appleScore = doc1Row[appleIdx];
    const bananaScore = doc1Row[bananaIdx];

    // Apple is in 1/2 docs (higher IDF), Banana in 2/2 (lower IDF)
    // TF(apple) = 2/3, TF(banana) = 1/3
    // IDF(apple) = log(2/1) ~= 0.69
    // IDF(banana) = log(2/2) = 0
    expect(appleScore).toBeGreaterThan(bananaScore);
    expect(bananaScore).toBe(0); // Because log(1) = 0
  });
});