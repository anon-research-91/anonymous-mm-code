# scripts/graph_delta/compat.py
import os

# ------- TMP 目录防止 torch/transformers 报错 -------
def setup_tmpdir():
    if not os.path.exists("/tmp"):
        os.makedirs("/tmp", mode=0o1777, exist_ok=True)
    if not os.access("/tmp", os.W_OK):
        project_tmp = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp")
        os.makedirs(project_tmp, exist_ok=True)
        os.environ["TMPDIR"] = project_tmp
        os.environ["TEMP"] = project_tmp
        os.environ["TMP"] = project_tmp


def print_key(msg: str):
    print(msg, flush=True)
