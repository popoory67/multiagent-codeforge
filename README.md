# Architecture Overview
project/
 ├─ prompts/
 │   └─ prompts.yaml
 ├─ config/
 │   │  models.yaml
 │   └─ config.yaml
 ├─ multi_agents_qml.py
 └─ README.md

## Processing
Agent.generate() / Agent.lint() / Agent.static_fix() ...
      ↓
BaseAgent.chat(system, user)
      ↓
LLMClient.chat(messages, ...)
      ↓
OpenAI streaming

## Prerequisite
```
pip install --upgrade autogen
```

If the compatibility in your environment didn't work well, run these command below which remove your python packages and reinstall correct version of `autogen`
```
pip uninstall -y autogen
pip uninstall -y pyautogen
pip uninstall -y autogen-agentchat
pip uninstall -y autogen-core
pip uninstall -y autogen-ext
pip uninstall -y autogenstudio
pip uninstall -y ag2

pip install --upgrade autogen
```
