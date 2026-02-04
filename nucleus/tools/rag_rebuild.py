#!/usr/bin/env python3
"""
RAG Rebuild Tool
================
Wrapper around rag_vector.py to rebuild the index from events.ndjson.
Ensures the vector store is fully synchronized with the System of Record (SoR).
"""
import sys
import os
import json
from pathlib import Path

# Add current directory to path to find rag_vector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rag_vector import VectorRAG, DB_PATH
except ImportError:
    # Try creating a stub if imported from elsewhere or if path issues
    sys.stderr.write("Error: Could not import rag_vector.py. Ensure it is in the same directory.\n")
    sys.exit(1)

EVENTS_PATH = "/pluribus/.pluribus/bus/events.ndjson"

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python3 rag_rebuild.py [events_path]")
        sys.exit(0)

    events_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(EVENTS_PATH)

    if not events_path.exists():
        print(f"Error: Events file not found at {events_path}")
        sys.exit(1)

    print(f"Rebuilding RAG index from {events_path}...")
    
    # Initialize RAG
    rag = VectorRAG(Path(DB_PATH), load_model=True)
    
    try:
        # Check vector availability
        stats = rag.stats()
        if not stats.get("vec_available"):
            print("Warning: sqlite-vec not available. Falling back to BM25 only.")
        
        # Rebuild
        rag.init_schema()
        result = rag.rebuild_from_bus(events_path)
        
        print(json.dumps(result, indent=2))
        
        if result.get("errors", 0) > 0:
            print(f"Completed with {result['errors']} errors.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error rebuilding index: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        rag.close()

if __name__ == "__main__":
    main()
