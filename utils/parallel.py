from concurrent.futures import ThreadPoolExecutor

def run_agents_in_parallel(agent_pipelines, max_workers=3):
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(p.run) for p in agent_pipelines]
        results = [f.result() for f in futures]
    return results