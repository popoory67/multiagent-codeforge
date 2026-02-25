# main.py
from pathlib import Path
from utils.llm_client import LLMClient
from utils.project_utils import summarize_project
from utils.diff_utils import is_unified_diff, apply_unified_diff
from agents.generator_agent import GenerateAgent
from agents.linter_agent import LinterAgent
from agents.reviewer_agent import ReviewerAgent
import yaml, os

def load_yaml(p): return yaml.safe_load(open(p, "r", encoding="utf-8"))

def main():
    script = Path(__file__).resolve().parent
    prompts = load_yaml(script / "config" / "prompts.yaml")
    config  = load_yaml(script / "config" / "config.yaml")
    models  = load_yaml(script / "config" / "models.yaml")

    mname = models["default_model"]
    mcfg  = models["models"][mname]

    llm = LLMClient(
        base_url=mcfg["api_base"],
        api_key=mcfg["api_key"],
        model=mcfg.get("model", mname),
        temperature=mcfg.get("temperature", 0.2)
    )

    # 1. Summarize
    ctx = summarize_project(".", tuple(config["project"]["target_exts"]), max_chars=9000)

    # 2. Generate
    gen_agent = GenerateAgent(llm, prompts)
    gen = gen_agent.generate(ctx)
    print("=== GENERATE ===\n", gen)

    if not is_unified_diff(gen):
        print("[ERR] not unified diff")
        return

    # 3. Lint + static fix
    lint_agent = LinterAgent(llm, prompts, config)
    lint = lint_agent.apply_and_lint(gen)
    print("\n=== LINT ===\n", lint)

    static_fix = lint_agent.static_fix(lint)
    print("\n=== STATIC ===\n", static_fix)

    # 4. Review
    reviewer = ReviewerAgent(llm, prompts)
    final = reviewer.review(gen, lint, static_fix)
    print("\n=== FINAL ===\n", final)

    if config["options"]["apply_patch"] and is_unified_diff(final):
        applied = apply_unified_diff(final)
        if applied.get("applied"):
            os.system("git add -A")
            os.system('git commit -m "agent: apply final QML patch"')
            print("[OK] committed")
        else:
            print("[ERR] apply failed:", applied)

if __name__ == "__main__":
    main()