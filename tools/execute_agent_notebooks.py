from __future__ import annotations

import ast
import contextlib
import io
from pathlib import Path

import nbformat
from nbformat.v4 import new_output

ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT/"notebooks/07_knowledge_graph_rag.ipynb", ROOT/"notebooks/08_multi_agent_system.ipynb"]:
    nb = nbformat.read(path, as_version=4)
    env = {"__name__": "__main__"}
    count = 0
    for cell in nb.cells:
        if cell.cell_type != "code": continue
        count += 1; cell.execution_count = count; cell.outputs = []
        tree = ast.parse(cell.source); last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr): last = ast.Expression(tree.body.pop().value)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            if tree.body: exec(compile(tree, str(path), "exec"), env)
            value = eval(compile(last, str(path), "eval"), env) if last else None
        if stream.getvalue(): cell.outputs.append(new_output("stream", name="stdout", text=stream.getvalue()))
        if value is not None: cell.outputs.append(new_output("execute_result", execution_count=count, data={"text/plain": repr(value)}, metadata={}))
    nbformat.write(nb, path)
    print(path.name, count, "code cells")
