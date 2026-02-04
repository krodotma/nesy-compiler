#!/usr/bin/env python3
"""
typescript_transformer.py - TypeScript Code Transformer (Step 55)

PBTSO Phase: ITERATE

Provides:
- TypeScript AST transformations via subprocess to ts-morph
- Interface and type manipulation
- Import/export handling
- Fallback regex-based transformations

Bus Topics:
- code.transform.typescript
- code.transform.complete

Protocol: DKIN v30

Note: Full AST support requires Node.js with ts-morph installed.
This module provides both subprocess-based ts-morph integration
and regex-based fallbacks for environments without Node.js.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Types
# =============================================================================

@dataclass
class TSTransformResult:
    """Result of a TypeScript transformation."""
    success: bool
    source: str
    transformed: str
    changes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    used_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "source_len": len(self.source),
            "transformed_len": len(self.transformed),
            "changes": self.changes,
            "warnings": self.warnings,
            "elapsed_ms": self.elapsed_ms,
            "used_fallback": self.used_fallback,
        }


@dataclass
class TSImport:
    """Represents a TypeScript import."""
    module: str
    default_import: Optional[str] = None
    named_imports: List[str] = field(default_factory=list)
    namespace_import: Optional[str] = None
    is_type_only: bool = False

    def to_source(self) -> str:
        """Convert to TypeScript import statement."""
        parts = []

        type_prefix = "type " if self.is_type_only else ""

        if self.default_import:
            parts.append(self.default_import)

        if self.named_imports:
            named = ", ".join(self.named_imports)
            parts.append(f"{{ {named} }}")

        if self.namespace_import:
            parts.append(f"* as {self.namespace_import}")

        if parts:
            return f"import {type_prefix}{', '.join(parts)} from '{self.module}';"
        else:
            return f"import '{self.module}';"


# =============================================================================
# TypeScript Code Transformer
# =============================================================================

class TypeScriptCodeTransformer:
    """
    TypeScript AST transformations via ts-morph or regex fallback.

    PBTSO Phase: ITERATE

    Provides methods for:
    - Adding imports
    - Adding/modifying interfaces
    - Adding/modifying types
    - Renaming symbols
    - Adding/removing decorators

    Note: For full AST support, ensure ts-morph is installed:
        npm install ts-morph
    """

    def __init__(
        self,
        bus: Optional[Any] = None,
        timeout_s: int = 30,
        prefer_fallback: bool = False,
    ):
        self.bus = bus
        self.timeout_s = timeout_s
        self.prefer_fallback = prefer_fallback
        self._changes: List[str] = []
        self._warnings: List[str] = []
        self._ts_morph_available: Optional[bool] = None

    def transform(
        self,
        source: str,
        transformations: List[Dict[str, Any]],
    ) -> TSTransformResult:
        """
        Apply transformations to TypeScript source.

        Args:
            source: TypeScript source code
            transformations: List of transformation specs

        Returns:
            TSTransformResult with transformed code
        """
        self._changes = []
        self._warnings = []
        start_time = time.time()

        if not self.prefer_fallback and self._check_ts_morph():
            try:
                result = self._transform_with_ts_morph(source, transformations)
                if result.success:
                    return result
                # Fall through to fallback on failure
            except Exception as e:
                self._warnings.append(f"ts-morph failed: {e}")

        # Use regex-based fallback
        return self._transform_with_fallback(source, transformations, start_time)

    # =========================================================================
    # High-Level Operations
    # =========================================================================

    def add_import(
        self,
        source: str,
        module: str,
        default_import: Optional[str] = None,
        named_imports: Optional[List[str]] = None,
        is_type_only: bool = False,
    ) -> TSTransformResult:
        """
        Add an import statement.

        Args:
            source: TypeScript source
            module: Module to import from
            default_import: Default import name
            named_imports: Named imports
            is_type_only: Whether this is a type-only import

        Returns:
            TSTransformResult with modified source
        """
        ts_import = TSImport(
            module=module,
            default_import=default_import,
            named_imports=named_imports or [],
            is_type_only=is_type_only,
        )

        return self.transform(source, [{
            "type": "add_import",
            "import": ts_import,
        }])

    def add_interface(
        self,
        source: str,
        name: str,
        properties: Dict[str, str],
        extends: Optional[List[str]] = None,
        export: bool = True,
    ) -> TSTransformResult:
        """
        Add an interface definition.

        Args:
            source: TypeScript source
            name: Interface name
            properties: Property name -> type mapping
            extends: Interfaces to extend
            export: Whether to export the interface

        Returns:
            TSTransformResult with modified source
        """
        return self.transform(source, [{
            "type": "add_interface",
            "name": name,
            "properties": properties,
            "extends": extends or [],
            "export": export,
        }])

    def add_type_alias(
        self,
        source: str,
        name: str,
        type_definition: str,
        export: bool = True,
    ) -> TSTransformResult:
        """
        Add a type alias.

        Args:
            source: TypeScript source
            name: Type name
            type_definition: Type definition
            export: Whether to export the type

        Returns:
            TSTransformResult with modified source
        """
        return self.transform(source, [{
            "type": "add_type",
            "name": name,
            "definition": type_definition,
            "export": export,
        }])

    def rename_symbol(
        self,
        source: str,
        old_name: str,
        new_name: str,
    ) -> TSTransformResult:
        """
        Rename a symbol throughout the source.

        Args:
            source: TypeScript source
            old_name: Current name
            new_name: New name

        Returns:
            TSTransformResult with modified source
        """
        return self.transform(source, [{
            "type": "rename",
            "old_name": old_name,
            "new_name": new_name,
        }])

    # =========================================================================
    # ts-morph Integration
    # =========================================================================

    def _check_ts_morph(self) -> bool:
        """Check if ts-morph is available."""
        if self._ts_morph_available is not None:
            return self._ts_morph_available

        try:
            result = subprocess.run(
                ["npx", "ts-morph", "--version"],
                capture_output=True,
                timeout=5,
            )
            self._ts_morph_available = result.returncode == 0
        except Exception:
            self._ts_morph_available = False

        return self._ts_morph_available

    def _transform_with_ts_morph(
        self,
        source: str,
        transformations: List[Dict[str, Any]],
    ) -> TSTransformResult:
        """Transform using ts-morph subprocess."""
        start_time = time.time()

        # Build the transform script
        script = self._build_transform_script(source, transformations)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            result = subprocess.run(
                ["npx", "ts-node", "--transpile-only", script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )

            if result.returncode != 0:
                return TSTransformResult(
                    success=False,
                    source=source,
                    transformed=source,
                    changes=[],
                    warnings=[f"ts-morph error: {result.stderr}"],
                    elapsed_ms=(time.time() - start_time) * 1000,
                )

            transformed = result.stdout.strip()

            return TSTransformResult(
                success=True,
                source=source,
                transformed=transformed,
                changes=self._changes,
                warnings=self._warnings,
                elapsed_ms=(time.time() - start_time) * 1000,
                used_fallback=False,
            )

        finally:
            try:
                os.unlink(script_path)
            except Exception:
                pass

    def _build_transform_script(
        self,
        source: str,
        transformations: List[Dict[str, Any]],
    ) -> str:
        """Build a ts-morph transformation script."""
        # Escape the source for template literal
        escaped_source = source.replace("`", "\\`").replace("$", "\\$")

        transform_code = []
        for t in transformations:
            t_type = t.get("type")

            if t_type == "add_import":
                imp = t.get("import", {})
                if isinstance(imp, TSImport):
                    imp = {
                        "module": imp.module,
                        "default_import": imp.default_import,
                        "named_imports": imp.named_imports,
                        "is_type_only": imp.is_type_only,
                    }
                transform_code.append(f"""
sourceFile.addImportDeclaration({{
    moduleSpecifier: '{imp.get("module", "")}',
    {'defaultImport: "' + imp["default_import"] + '",' if imp.get("default_import") else ''}
    {'namedImports: ' + json.dumps(imp.get("named_imports", [])) + ',' if imp.get("named_imports") else ''}
    {'isTypeOnly: true,' if imp.get("is_type_only") else ''}
}});
""")

            elif t_type == "add_interface":
                props = t.get("properties", {})
                prop_code = ", ".join([
                    f'{{ name: "{k}", type: "{v}" }}'
                    for k, v in props.items()
                ])
                transform_code.append(f"""
sourceFile.addInterface({{
    name: '{t.get("name", "NewInterface")}',
    isExported: {str(t.get("export", True)).lower()},
    properties: [{prop_code}],
}});
""")

            elif t_type == "rename":
                transform_code.append(f"""
const declarations = sourceFile.getDescendantsOfKind(ts.SyntaxKind.Identifier)
    .filter(id => id.getText() === '{t.get("old_name", "")}');
for (const decl of declarations) {{
    decl.replaceWithText('{t.get("new_name", "")}');
}}
""")

        return f'''
import {{ Project, ts }} from "ts-morph";

const project = new Project({{ useInMemoryFileSystem: true }});
const sourceFile = project.createSourceFile("temp.ts", `{escaped_source}`);

{chr(10).join(transform_code)}

console.log(sourceFile.getFullText());
'''

    # =========================================================================
    # Regex-Based Fallback
    # =========================================================================

    def _transform_with_fallback(
        self,
        source: str,
        transformations: List[Dict[str, Any]],
        start_time: float,
    ) -> TSTransformResult:
        """Transform using regex-based fallback."""
        transformed = source

        for t in transformations:
            t_type = t.get("type")

            if t_type == "add_import":
                transformed = self._fallback_add_import(transformed, t)

            elif t_type == "add_interface":
                transformed = self._fallback_add_interface(transformed, t)

            elif t_type == "add_type":
                transformed = self._fallback_add_type(transformed, t)

            elif t_type == "rename":
                transformed = self._fallback_rename(transformed, t)

        return TSTransformResult(
            success=True,
            source=source,
            transformed=transformed,
            changes=self._changes,
            warnings=self._warnings,
            elapsed_ms=(time.time() - start_time) * 1000,
            used_fallback=True,
        )

    def _fallback_add_import(self, source: str, transform: Dict[str, Any]) -> str:
        """Add import using regex (fallback)."""
        imp = transform.get("import", {})
        if isinstance(imp, TSImport):
            import_statement = imp.to_source()
        else:
            # Build from dict
            parts = []
            if imp.get("default_import"):
                parts.append(imp["default_import"])
            if imp.get("named_imports"):
                named = ", ".join(imp["named_imports"])
                parts.append(f"{{ {named} }}")

            module = imp.get("module", "")
            type_prefix = "type " if imp.get("is_type_only") else ""

            if parts:
                import_statement = f"import {type_prefix}{', '.join(parts)} from '{module}';"
            else:
                import_statement = f"import '{module}';"

        # Find position for import (after existing imports or at top)
        import_pattern = r"^import\s+.*?;?\s*$"
        last_import_match = None
        for match in re.finditer(import_pattern, source, re.MULTILINE):
            last_import_match = match

        if last_import_match:
            insert_pos = last_import_match.end()
            source = source[:insert_pos] + "\n" + import_statement + source[insert_pos:]
        else:
            source = import_statement + "\n\n" + source

        self._changes.append(f"Added import from '{imp.get('module', '')}'")
        return source

    def _fallback_add_interface(self, source: str, transform: Dict[str, Any]) -> str:
        """Add interface using regex (fallback)."""
        name = transform.get("name", "NewInterface")
        properties = transform.get("properties", {})
        extends = transform.get("extends", [])
        export = transform.get("export", True)

        # Build interface
        export_keyword = "export " if export else ""
        extends_clause = f" extends {', '.join(extends)}" if extends else ""

        prop_lines = []
        for prop_name, prop_type in properties.items():
            prop_lines.append(f"  {prop_name}: {prop_type};")

        interface_code = f"""
{export_keyword}interface {name}{extends_clause} {{
{chr(10).join(prop_lines)}
}}
"""

        # Append at end of file
        source = source.rstrip() + "\n" + interface_code

        self._changes.append(f"Added interface '{name}'")
        return source

    def _fallback_add_type(self, source: str, transform: Dict[str, Any]) -> str:
        """Add type alias using regex (fallback)."""
        name = transform.get("name", "NewType")
        definition = transform.get("definition", "unknown")
        export = transform.get("export", True)

        export_keyword = "export " if export else ""
        type_code = f"\n{export_keyword}type {name} = {definition};\n"

        # Append at end of file
        source = source.rstrip() + type_code

        self._changes.append(f"Added type alias '{name}'")
        return source

    def _fallback_rename(self, source: str, transform: Dict[str, Any]) -> str:
        """Rename symbol using regex (fallback)."""
        old_name = transform.get("old_name", "")
        new_name = transform.get("new_name", "")

        if not old_name or not new_name:
            return source

        # Use word boundary matching
        pattern = rf"\b{re.escape(old_name)}\b"
        count = len(re.findall(pattern, source))
        source = re.sub(pattern, new_name, source)

        self._changes.append(f"Renamed '{old_name}' to '{new_name}' ({count} occurrences)")
        return source

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def parse_imports(self, source: str) -> List[TSImport]:
        """Parse import statements from source."""
        imports = []

        # Match various import patterns
        patterns = [
            # import X from 'module'
            r"import\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]",
            # import { X, Y } from 'module'
            r"import\s+\{([^}]+)\}\s+from\s+['\"]([^'\"]+)['\"]",
            # import * as X from 'module'
            r"import\s+\*\s+as\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]",
            # import 'module'
            r"import\s+['\"]([^'\"]+)['\"]",
        ]

        for match in re.finditer(patterns[0], source):
            imports.append(TSImport(
                module=match.group(2),
                default_import=match.group(1),
            ))

        for match in re.finditer(patterns[1], source):
            named = [n.strip() for n in match.group(1).split(",")]
            imports.append(TSImport(
                module=match.group(2),
                named_imports=named,
            ))

        for match in re.finditer(patterns[2], source):
            imports.append(TSImport(
                module=match.group(2),
                namespace_import=match.group(1),
            ))

        for match in re.finditer(patterns[3], source):
            imports.append(TSImport(module=match.group(1)))

        return imports

    def get_exported_names(self, source: str) -> List[Tuple[str, str]]:
        """Get exported names and their types (function, class, interface, type, const)."""
        exports = []

        patterns = [
            (r"export\s+function\s+(\w+)", "function"),
            (r"export\s+class\s+(\w+)", "class"),
            (r"export\s+interface\s+(\w+)", "interface"),
            (r"export\s+type\s+(\w+)", "type"),
            (r"export\s+const\s+(\w+)", "const"),
            (r"export\s+let\s+(\w+)", "let"),
            (r"export\s+enum\s+(\w+)", "enum"),
        ]

        for pattern, kind in patterns:
            for match in re.finditer(pattern, source):
                exports.append((match.group(1), kind))

        return exports
