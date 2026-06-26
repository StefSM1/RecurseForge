"""
context/repo_map.py
===================
Tree-sitter repository map server (FastAPI).

Parses a target codebase into an AST index and serves surgical code
lookups to sub-agents via three endpoints:

    GET  /map      -- full repo map (XML-packed, configurable token budget)
    POST /lookup   -- specific file / symbol / line range
    POST /refresh  -- re-parse after file changes

Run standalone:
    python -m context.repo_map --target-dir ../my-project --port 8001
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import tiktoken
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
from fastapi import FastAPI, HTTPException
from tree_sitter import Language, Parser
import uvicorn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from engine.interfaces import ContextRequest, ContextPayload

logger = logging.getLogger("recurseforge.context.repo_map")

PY_LANGUAGE = Language(tspython.language())
JS_LANGUAGE = Language(tsjavascript.language())
LANG_MAP = {".py": PY_LANGUAGE, ".js": JS_LANGUAGE, ".ts": JS_LANGUAGE}

SKIP_DIRS = {
    ".venv", "venv", "__pycache__", "node_modules", ".git",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
}

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


class SymbolInfo:
    __slots__ = ("kind", "name", "signature", "start_line", "end_line")

    def __init__(self, kind, name, signature, start_line, end_line):
        self.kind = kind
        self.name = name
        self.signature = signature
        self.start_line = start_line
        self.end_line = end_line

    def to_xml(self):
        if self.kind == "class":
            tag = 'class name="{}"'.format(self.name)
            body = "line {}-{}".format(self.start_line, self.end_line)
            return "      <{t}>{b}</{t}>".format(t=tag, b=body)
        tag = 'function name="{}" sig="{}"'.format(self.name, self.signature)
        body = "line {}-{}".format(self.start_line, self.end_line)
        return "      <{t}>{b}</{t}>".format(t=tag, b=body)


class FileIndex:
    __slots__ = ("path", "symbols", "source")

    def __init__(self, path, symbols, source):
        self.path = path
        self.symbols = symbols
        self.source = source


class RepoMap:
    """In-memory AST index of a target codebase."""

    def __init__(self, target_dir: str):
        self.target_dir = Path(target_dir).resolve()
        self._parsers: dict[str, Parser] = {}
        for ext, lang in LANG_MAP.items():
            p = Parser(lang)
            self._parsers[ext] = p
        self._index: dict[str, FileIndex] = {}
        self.parse_all()

    def _get_parser(self, ext: str):
        return self._parsers.get(ext)

    def parse_all(self):
        self._index.clear()
        count = 0
        for root, dirs, files in os.walk(self.target_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in files:
                ext = Path(fname).suffix
                if ext not in LANG_MAP:
                    continue
                fpath = Path(root) / fname
                try:
                    self._parse_file(fpath)
                    count += 1
                except Exception as e:
                    logger.warning("Failed to parse %s: %s", fpath, e)
        logger.info("[RepoMap] Parsed %d files from %s", count, self.target_dir)

    def _parse_file(self, fpath: Path):
        ext = fpath.suffix
        parser = self._get_parser(ext)
        if parser is None:
            return
        source = fpath.read_text(encoding="utf-8", errors="replace")
        tree = parser.parse(source.encode("utf-8"))
        symbols = self._extract_symbols(tree.root_node, source, ext)
        rel_path = str(fpath.relative_to(self.target_dir))
        self._index[rel_path] = FileIndex(rel_path, symbols, source)

    def _extract_symbols(self, root_node, source: str, ext: str):
        symbols = []
        for child in root_node.children:
            ntype = child.type
            if ext == ".py":
                if ntype == "class_definition":
                    name = self._child_text(child, "identifier") or "?"
                    symbols.append(SymbolInfo(
                        "class", name, "",
                        child.start_point[0] + 1, child.end_point[0] + 1))
                elif ntype == "function_definition":
                    name = self._child_text(child, "identifier") or "?"
                    sig = self._extract_py_params(child)
                    symbols.append(SymbolInfo(
                        "function", name, sig,
                        child.start_point[0] + 1, child.end_point[0] + 1))
            elif ext in (".js", ".ts"):
                if ntype in ("function_declaration", "method_definition"):
                    name = self._child_text(child, "identifier") or "?"
                    symbols.append(SymbolInfo(
                        "function", name, "",
                        child.start_point[0] + 1, child.end_point[0] + 1))
                elif ntype == "class_declaration":
                    name = self._child_text(child, "identifier") or "?"
                    symbols.append(SymbolInfo(
                        "class", name, "",
                        child.start_point[0] + 1, child.end_point[0] + 1))
        return symbols

    def _child_text(self, node, child_type: str) -> str:
        for c in node.children:
            if c.type == child_type:
                return c.text.decode("utf-8") if c.text else ""
        return ""

    def _extract_py_params(self, func_node) -> str:
        for c in func_node.children:
            if c.type == "parameters":
                return c.text.decode("utf-8") if c.text else "()"
        return "()"

    def generate_map(self, max_tokens: int = 4096) -> str:
        lines = ["<codebase_summary>"]
        for fpath in sorted(self._index.keys()):
            fi = self._index[fpath]
            if not fi.symbols:
                continue
            lines.append('    <file path="{}">'.format(fpath))
            for sym in fi.symbols:
                lines.append(sym.to_xml())
            lines.append("    </file>")
        lines.append("</codebase_summary>")
        result = "\n".join(lines)
        if count_tokens(result) > max_tokens:
            lines_trunc = ["<codebase_summary>"]
            for fpath in sorted(self._index.keys()):
                fi = self._index[fpath]
                if not fi.symbols:
                    continue
                lines_trunc.append('    <file path="{}">'.format(fpath))
                for sym in fi.symbols[:8]:
                    lines_trunc.append(sym.to_xml())
                if len(fi.symbols) > 8:
                    lines_trunc.append(
                        "      <!-- {} more symbols -->".format(
                            len(fi.symbols) - 8))
                lines_trunc.append("    </file>")
                partial = "\n".join(lines_trunc + ["</codebase_summary>"])
                if count_tokens(partial) > max_tokens:
                    break
            result = "\n".join(lines_trunc + ["</codebase_summary>"])
        return result

    def lookup(self, req: ContextRequest) -> ContextPayload:
        # Normalize path separators for cross-platform compatibility
        normalized = req.file_path.replace("/", os.sep).replace("\\", os.sep)
        fi = self._index.get(normalized) or self._index.get(req.file_path)
        if fi is None:
            return ContextPayload(
                node_id=req.node_id, file_path=req.file_path,
                content="ERROR: file not found in index", token_count=0)
        if req.line_range:
            start, end = req.line_range
            src_lines = fi.source.splitlines()
            start = max(1, start)
            end = min(len(src_lines), end)
            content = "\n".join(src_lines[start - 1:end])
        elif req.symbol_name:
            content = self._find_symbol(fi, req.symbol_name)
        else:
            content = fi.source
        return ContextPayload(
            node_id=req.node_id, file_path=req.file_path,
            content=content, token_count=count_tokens(content))

    def _find_symbol(self, fi: FileIndex, name: str) -> str:
        for sym in fi.symbols:
            if sym.name == name:
                src_lines = fi.source.splitlines()
                return "\n".join(
                    src_lines[sym.start_line - 1:sym.end_line])
        return "ERROR: symbol '{}' not found in {}".format(name, fi.path)


_REPO_MAP_INSTANCE = None
app = FastAPI(title="RecurseForge Repo-Map Server")


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------

@app.get("/map")
def get_repo_map(max_tokens: int = 4096):
    """Return the full repository map as XML-packed text."""
    if _REPO_MAP_INSTANCE is None:
        raise HTTPException(500, "RepoMap not initialized")
    xml = _REPO_MAP_INSTANCE.generate_map(max_tokens=max_tokens)
    return {
        "map": xml,
        "token_count": count_tokens(xml),
        "files_indexed": len(_REPO_MAP_INSTANCE._index),
    }


@app.post("/lookup")
def post_lookup(req: ContextRequest):
    """Return a specific code fragment for a file/symbol/line range."""
    if _REPO_MAP_INSTANCE is None:
        raise HTTPException(500, "RepoMap not initialized")
    payload = _REPO_MAP_INSTANCE.lookup(req)
    return payload.model_dump()


@app.post("/refresh")
def post_refresh():
    """Re-parse the codebase (call after agents modify files)."""
    if _REPO_MAP_INSTANCE is None:
        raise HTTPException(500, "RepoMap not initialized")
    _REPO_MAP_INSTANCE.parse_all()
    return {
        "status": "ok",
        "files_indexed": len(_REPO_MAP_INSTANCE._index),
    }


@app.get("/health")
def health():
    return {"status": "ok", "initialized": _REPO_MAP_INSTANCE is not None}


# ---------------------------------------------------------------------------
# Startup / CLI
# ---------------------------------------------------------------------------

def create_app(target_dir: str, max_tokens: int = 4096) -> FastAPI:
    """Create and configure the FastAPI app with a RepoMap instance."""
    global _REPO_MAP_INSTANCE
    _REPO_MAP_INSTANCE = RepoMap(target_dir)
    logger.info("[RepoMap] Server ready. Target: %s (%d files indexed)",
                target_dir, len(_REPO_MAP_INSTANCE._index))
    return app


def main():
    parser = argparse.ArgumentParser(description="RecurseForge Repo-Map Server")
    parser.add_argument("--target-dir", required=True,
                        help="Directory to parse and serve")
    parser.add_argument("--port", type=int, default=8001,
                        help="Port to listen on (default: 8001)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    create_app(target_dir=args.target_dir, max_tokens=args.max_tokens)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
