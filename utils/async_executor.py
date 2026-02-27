# utils/async_executor.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any


async def run_async_jobs(jobs, workers: int = 4):
    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = [
            loop.run_in_executor(pool, job.run)
            for job in jobs
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

    final = []
    for job, res in zip(jobs, results):
        if isinstance(res, Exception):
            final.append({"id": job.id, "error": str(res)})
        else:
            final.append(res)

    return final