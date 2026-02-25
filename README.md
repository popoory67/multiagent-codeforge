# Architecture Overview
project/
 ├─ prompts/
 │   └─ prompts.yaml
 ├─ config/
 │   │  models.yaml
 │   └─ config.yaml
 ├─ multi_agents_qml.py
 └─ README.md

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
 
# Multi-Agent QML Refactoring Pipeline

본 프로젝트는 **3단계 Multi-Agent LLM Pipeline**으로 QML 프로젝트를 자동 수정, 정적 분석, 리뷰까지 수행합니다.

## 구성요소

### 1) QMLCoder (코드 생성)
- 요구사항(TASK_DESCRIPTION)에 따라 필요한 QML 패치를 생성
- output: unified diff only

### 2) QMLStaticAnalyzer (qmllint 기반 오류 수정)
- qmllint 출력 분석
- 보완 패치만 diff로 출력

### 3) Reviewer
- QMLCoder, StaticAnalyzer, Fitness Criteria를 종합
- 최종 승인 패치를 diff로 출력

---

## 사용 도구

### qmllint (Qt 공식 QML 정적 분석기)
- QML 문법 오류
- import 검증
- deprecated API 감지

### LLM
- OpenAI 호환 API / Ollama 기반 llama3.1:8b

---

## 설치

```bash
pip install autogen unidiff pyyaml