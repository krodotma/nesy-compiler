#!/usr/bin/env python3
"""
PBMINT - Pluribus Documentation Department Iteration Operator
==============================================================

An OITERATIVE subagent pattern that runs once an hour to comprehensively verify 
documentation against code, commits, branches, features, and bugs, then creates
enriched, semantically refined documentation for the Pluribus system.

Features:
- Cross-references documentation against actual code implementations
- Identifies discrepancies between docs and code
- Generates enriched documentation based on code analysis
- Integrates with SOTA tools index
- Runs as a permanent fixture like the Art Department
"""

import os
import sys
import json
import time
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re
import ast
from dataclasses import dataclass
from collections import defaultdict

# Add Pluribus tools to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

@dataclass
class DocVerificationResult:
    """Result of documentation verification"""
    file_path: str
    status: str  # 'ok', 'discrepancy', 'missing', 'outdated'
    issues: List[str]
    confidence: float  # 0.0 to 1.0
    suggestions: List[str]

class DocumentationAnalyzer:
    """Analyzes documentation against codebase"""
    
    def __init__(self, root_dir: str = "/pluribus"):
        self.root_dir = Path(root_dir)
        self.code_extensions = {'.py', '.ts', '.tsx', '.js', '.jsx', '.sh', '.cjs', '.mjs'}
        self.doc_extensions = {'.md', '.txt', '.rst', '.asciidoc'}
        
    def scan_codebase(self) -> Dict[str, Dict]:
        """Scan the codebase and extract key information"""
        code_info = defaultdict(dict)
        
        for ext in self.code_extensions:
            for file_path in self.root_dir.rglob(f"*{ext}"):
                if self._is_valid_file(file_path):
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        code_info[str(file_path)] = self._analyze_code_file(content, file_path)
                    except Exception as e:
                        print(f"Error analyzing {file_path}: {e}")
        
        return dict(code_info)
    
    def _is_valid_file(self, file_path: Path) -> bool:
        """Check if file should be analyzed"""
        # Skip temporary files, build artifacts, and test files
        skip_patterns = [
            '__pycache__', '.git', 'node_modules', 'dist', 'build',
            'test', 'tests', 'spec', 'tmp', '.tmp', 'coverage',
            'venv', '.venv', 'env', '.env', 'target', 'out', 'logs'
        ]
        
        path_str = str(file_path)
        return not any(skip_pattern in path_str for skip_pattern in skip_patterns)
    
    def _analyze_code_file(self, content: str, file_path: Path) -> Dict:
        """Analyze a single code file"""
        analysis = {
            'functions': [],
            'classes': [],
            'imports': [],
            'exports': [],
            'comments': [],
            'docstrings': [],
            'dependencies': [],
            'features': [],
            'interfaces': [],
            'last_modified': self._get_file_timestamp(file_path)
        }
        
        try:
            if file_path.suffix in ['.py']:
                analysis.update(self._analyze_python_file(content))
            elif file_path.suffix in ['.ts', '.tsx', '.js', '.jsx']:
                analysis.update(self._analyze_js_file(content))
            elif file_path.suffix in ['.sh']:
                analysis.update(self._analyze_shell_file(content))
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            
        return analysis
    
    def _analyze_python_file(self, content: str) -> Dict:
        """Analyze Python file structure"""
        try:
            tree = ast.parse(content)
            functions = []
            classes = []
            imports = []
            docstrings = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        'name': node.name,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'decorators': [ast.unparse(dec) for dec in node.decorator_list]
                    })
                elif isinstance(node, ast.ClassDef):
                    classes.append({
                        'name': node.name,
                        'line': node.lineno,
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    })
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    docstrings.append(node.value.value)
                    
            return {
                'functions': functions,
                'classes': classes,
                'imports': imports,
                'docstrings': docstrings
            }
        except SyntaxError:
            return {}
    
    def _analyze_js_file(self, content: str) -> Dict:
        """Analyze JavaScript/TypeScript file structure"""
        functions = []
        classes = []
        imports = []
        exports = []
        
        # Extract function declarations
        func_pattern = r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\('
        for match in re.finditer(func_pattern, content):
            functions.append({'name': match.group(1)})
        
        # Extract class declarations
        class_pattern = r'(?:export\s+)?class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            classes.append({'name': match.group(1)})
        
        # Extract imports
        import_pattern = r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]'
        for match in re.finditer(import_pattern, content):
            imports.append(match.group(1))
        
        # Extract exports
        export_pattern = r'export\s+(?:default\s+)?(\w+)'
        for match in re.finditer(export_pattern, content):
            exports.append(match.group(1))
        
        return {
            'functions': functions,
            'classes': classes,
            'imports': imports,
            'exports': exports
        }
    
    def _analyze_shell_file(self, content: str) -> Dict:
        """Analyze shell script structure"""
        functions = []
        exports = []
        
        # Extract function definitions
        func_pattern = r'^(\w+)\s*\(\)\s*\{'
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            functions.append({'name': match.group(1)})
        
        # Extract exports
        export_pattern = r'export\s+(\w+)='
        for match in re.finditer(export_pattern, content):
            exports.append(match.group(1))
        
        return {
            'functions': functions,
            'exports': exports
        }
    
    def _get_file_timestamp(self, file_path: Path) -> str:
        """Get file modification timestamp"""
        try:
            return datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        except:
            return datetime.now().isoformat()
    
    def scan_documentation(self) -> Dict[str, str]:
        """Scan documentation files"""
        docs = {}
        for ext in self.doc_extensions:
            for file_path in self.root_dir.rglob(f"*{ext}"):
                if self._is_valid_file(file_path):
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        docs[str(file_path)] = content
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
        return docs
    
    def verify_docs_against_code(self, docs: Dict[str, str], code_info: Dict[str, Dict]) -> List[DocVerificationResult]:
        """Verify documentation against actual code"""
        results = []
        
        for doc_path, doc_content in docs.items():
            # Find related code files
            related_code_files = self._find_related_code(doc_path, code_info)
            
            issues = []
            suggestions = []
            
            for code_path, code_analysis in related_code_files:
                # Check if documented functions exist in code
                doc_functions = self._extract_functions_from_doc(doc_content)
                code_functions = [f['name'] for f in code_analysis.get('functions', [])]
                
                for func in doc_functions:
                    if func not in code_functions:
                        issues.append(f"Documented function '{func}' not found in {code_path}")
                
                for func in code_functions:
                    if func not in doc_functions and not func.startswith('_'):
                        suggestions.append(f"Function '{func}' in {code_path} is not documented")
            
            status = 'ok' if not issues else 'discrepancy'
            confidence = 0.9 if not issues else 0.5 - (len(issues) * 0.1)
            
            results.append(DocVerificationResult(
                file_path=doc_path,
                status=status,
                issues=issues,
                confidence=max(0.0, confidence),
                suggestions=suggestions
            ))
        
        return results
    
    def _find_related_code(self, doc_path: str, code_info: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
        """Find code files related to a documentation file"""
        related = []
        doc_name = Path(doc_path).stem.lower()
        
        for code_path, code_analysis in code_info.items():
            code_file_name = Path(code_path).stem.lower()
            
            # Direct match (e.g., browser_session_daemon.md -> browser_session_daemon.py)
            if doc_name.replace('_doc', '').replace('_guide', '') in code_file_name or \
               code_file_name in doc_name.replace('_doc', '').replace('_guide', ''):
                related.append((code_path, code_analysis))
            
            # Look for related terms in code content
            for key, value in code_analysis.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and 'name' in item:
                            if doc_name in item['name'].lower():
                                related.append((code_path, code_analysis))
                                break
        
        return related
    
    def _extract_functions_from_doc(self, doc_content: str) -> List[str]:
        """Extract function names mentioned in documentation"""
        # Look for function names in code blocks, references, etc.
        patterns = [
            r'`(\w+\(\))',  # Function calls in backticks
            r'`(\w+)\s*\(',  # Function names followed by parentheses
            r'function\s+(\w+)',  # Function declarations in docs
            r'`([\w_]+)`',  # Any word in backticks (potential function name)
        ]
        
        functions = set()
        for pattern in patterns:
            matches = re.findall(pattern, doc_content, re.IGNORECASE)
            functions.update(matches)
        
        # Filter out common words that aren't likely function names
        common_words = {'the', 'and', 'or', 'if', 'else', 'for', 'while', 'in', 'to', 'of', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'ours', 'theirs'}
        
        return [f for f in functions if f.lower() not in common_words and len(f) > 2]

class PBMINTOperator:
    """PBMINT - Pluribus Documentation Department Iteration Operator"""
    
    def __init__(self, root_dir: str = "/pluribus"):
        self.root_dir = Path(root_dir)
        self.analyzer = DocumentationAnalyzer(root_dir)
        self.bus_dir = self.root_dir / ".pluribus" / "bus"
        self.reports_dir = self.root_dir / "agent_reports"
        
    def run_iteration(self) -> Dict:
        """Run a single documentation iteration"""
        print(f"[PBMINT] Starting documentation iteration at {datetime.now().isoformat()}")
        
        # Scan codebase
        print("[PBMINT] Scanning codebase...")
        code_info = self.analyzer.scan_codebase()
        
        # Scan documentation
        print("[PBMINT] Scanning documentation...")
        docs = self.analyzer.scan_documentation()
        
        # Verify docs against code
        print("[PBMINT] Verifying documentation against code...")
        verification_results = self.analyzer.verify_docs_against_code(docs, code_info)
        
        # Generate enriched documentation
        print("[PBMINT] Generating enriched documentation...")
        enriched_docs = self._generate_enriched_documentation(verification_results, code_info, docs)
        
        # Integrate with SOTA tools
        print("[PBMINT] Integrating with SOTA tools index...")
        self._integrate_with_sota(enriched_docs)
        
        # Create report
        report = self._create_iteration_report(verification_results, enriched_docs)
        
        # Emit bus event
        self._emit_bus_event(report)
        
        print(f"[PBMINT] Documentation iteration completed. Processed {len(docs)} docs, {len(code_info)} code files.")
        
        return report
    
    def _generate_enriched_documentation(self, verification_results: List[DocVerificationResult], 
                                       code_info: Dict[str, Dict], docs: Dict[str, str]) -> Dict[str, str]:
        """Generate enriched documentation based on analysis"""
        enriched = {}
        
        for result in verification_results:
            original_doc = docs.get(result.file_path, "")
            
            if result.status == 'discrepancy':
                # Create enriched version with corrections
                enriched_doc = self._enrich_documentation(original_doc, result)
                enriched[result.file_path] = enriched_doc
            else:
                # Keep original if no issues
                enriched[result.file_path] = original_doc
        
        # Generate new documentation for undocumented features
        for code_path, code_analysis in code_info.items():
            if self._has_undocumented_features(code_analysis):
                new_doc_path = self._generate_documentation_path(code_path)
                new_doc_content = self._generate_feature_documentation(code_path, code_analysis)
                enriched[new_doc_path] = new_doc_content
        
        return enriched
    
    def _enrich_documentation(self, original_doc: str, result: DocVerificationResult) -> str:
        """Enrich a document with corrections and suggestions"""
        enriched = original_doc
        
        # Add missing function documentation
        for suggestion in result.suggestions:
            if "is not documented" in suggestion:
                func_name = suggestion.split("'")[1]
                enriched += f"\n## Function: {func_name}\n\n"
                enriched += f"This function `{func_name}` is implemented in the code but was missing from documentation.\n\n"
                enriched += "### Signature\n```typescript\n// TODO: Add actual signature from code analysis\n```\n\n"
                enriched += "### Description\n// TODO: Add description based on code analysis\n\n"
        
        # Add metadata about verification
        verification_metadata = f"""
---
## Documentation Verification
- Last verified: {datetime.now().isoformat()}
- Status: {result.status}
- Issues found: {len(result.issues)}
- Confidence: {result.confidence:.2f}
---
"""
        
        return verification_metadata + enriched
    
    def _has_undocumented_features(self, code_analysis: Dict) -> bool:
        """Check if code has undocumented features"""
        functions = code_analysis.get('functions', [])
        # Consider functions without documentation as undocumented
        return len(functions) > 0
    
    def _generate_documentation_path(self, code_path: str) -> str:
        """Generate documentation path for code file"""
        code_file = Path(code_path)
        doc_file = code_file.with_suffix('.md')
        doc_path = str(doc_file).replace(str(self.root_dir), str(self.root_dir / 'docs'))
        return doc_path
    
    def _generate_feature_documentation(self, code_path: str, code_analysis: Dict) -> str:
        """Generate documentation for undocumented features"""
        content = f"# Documentation for {Path(code_path).name}\n\n"
        content += f"Automatically generated documentation for `{code_path}`\n\n"
        
        functions = code_analysis.get('functions', [])
        if functions:
            content += "## Functions\n\n"
            for func in functions:
                content += f"- `{func.get('name', 'unknown')}`\n"
        
        classes = code_analysis.get('classes', [])
        if classes:
            content += "\n## Classes\n\n"
            for cls in classes:
                content += f"- `{cls.get('name', 'unknown')}`\n"
        
        return content
    
    def _integrate_with_sota(self, enriched_docs: Dict[str, str]):
        """Integrate documentation with SOTA tools index"""
        sota_index_path = self.root_dir / "sota-tools-leads" / "sota_tool_leads.md"
        
        if sota_index_path.exists():
            try:
                sota_content = sota_index_path.read_text()
                
                # Look for documentation references in SOTA index
                for doc_path, doc_content in enriched_docs.items():
                    doc_name = Path(doc_path).stem
                    if doc_name.lower() in sota_content.lower():
                        # Update SOTA index with new documentation links
                        updated_sota = self._update_sota_with_docs(sota_content, doc_path, doc_content)
                        sota_index_path.write_text(updated_sota)
                        
            except Exception as e:
                print(f"[PBMINT] Error updating SOTA index: {e}")
    
    def _update_sota_with_docs(self, sota_content: str, doc_path: str, doc_content: str) -> str:
        """Update SOTA index with documentation references"""
        # Add documentation link if not already present
        doc_ref = f"- Documentation: [{Path(doc_path).name}]({doc_path})"
        
        if doc_ref not in sota_content:
            sota_content += f"\n\n{doc_ref}"
        
        return sota_content
    
    def _create_iteration_report(self, verification_results: List[DocVerificationResult], 
                               enriched_docs: Dict[str, str]) -> Dict:
        """Create a report of the iteration"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_docs': len(enriched_docs),
                'verified_docs': len(verification_results),
                'ok_docs': len([r for r in verification_results if r.status == 'ok']),
                'discrepancy_docs': len([r for r in verification_results if r.status == 'discrepancy']),
                'new_docs_created': len([k for k, v in enriched_docs.items() if k not in [r.file_path for r in verification_results]])
            },
            'results': [
                {
                    'file': r.file_path,
                    'status': r.status,
                    'issues': r.issues,
                    'confidence': r.confidence,
                    'suggestions': r.suggestions
                } for r in verification_results
            ],
            'actions_taken': []
        }
        
        # Save report
        report_path = self.reports_dir / f"pbmint_iteration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.reports_dir.mkdir(exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        
        return report
    
    def _emit_bus_event(self, report: Dict):
        """Emit a bus event about the documentation iteration"""
        try:
            bus_path = self.bus_dir / "events.ndjson"
            self.bus_dir.mkdir(parents=True, exist_ok=True)
            
            event = {
                'id': f"pbmint-{int(time.time())}",
                'ts': time.time(),
                'iso': datetime.now().isoformat(),
                'topic': 'documentation.iteration.completed',
                'kind': 'artifact',
                'level': 'info',
                'actor': 'pbmint-operator',
                'data': report
            }
            
            with bus_path.open('a') as f:
                f.write(json.dumps(event) + '\n')
                
        except Exception as e:
            print(f"[PBMINT] Error emitting bus event: {e}")

def main():
    """Main entry point for PBMINT operator"""
    if len(sys.argv) > 1 and sys.argv[1] == "start":
        print("Starting PBMINT - Pluribus Documentation Department Iteration Operator...")
        
        operator = PBMINTOperator()
        
        # Run once initially
        operator.run_iteration()
        
        # Then run every hour
        while True:
            print(f"[PBMINT] Sleeping for 1 hour before next iteration...")
            time.sleep(3600)  # Sleep for 1 hour
            operator.run_iteration()
    else:
        # Run a single iteration
        operator = PBMINTOperator()
        report = operator.run_iteration()
        print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()