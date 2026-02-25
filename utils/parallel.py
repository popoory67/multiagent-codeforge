# utils/parallel.py
# -*- coding: utf-8 -*-

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

def run_agents_in_parallel(pipelines, max_workers: int = 3) -> List[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_map = {pool.submit(p.run): p for p in pipelines}
        for fut in as_completed(fut_map):
            p = fut_map[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"id": p.id, "error": str(e), "gen": "", "lint": "", "static": "", "final": ""})

    results.sort(key=lambda r: r.get("id", 0))
    return results