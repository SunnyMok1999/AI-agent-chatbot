# Multi-Agent Function Calling Template (Starter)

Use this as a practical schema for manager-agent coordination.

## 1) Tool call envelope

```json
{
  "type": "tool_call",
  "name": "request_clarification",
  "arguments": {
    "to_agent": "GaussAgent",
    "question": "Is this step valid under non-constant mass?"
  }
}
```

## 2) Minimal shared tools

- `request_clarification(to_agent, question)`
- `cite_source(query, top_k)`
- `check_dimensions(equation)`
- `verify_proof_step(statement, assumptions)`
- `solve_subproblem(expression, variable)`

## 3) Agent response contract

Each expert should return either:

```json
{
  "type": "expert_answer",
  "agent": "NewtonAgent",
  "hint": "...",
  "steps": ["...", "..."],
  "pitfall": "...",
  "confidence": 0.82,
  "citations": ["source_a.pdf#p12"]
}
```

or a tool call envelope.

## 4) Manager synthesis contract

```json
{
  "type": "manager_synthesis",
  "final_answer": "...",
  "confidence": 0.86,
  "resolved_conflicts": ["..."],
  "citations": ["source_a.pdf#p12", "source_b.md#sec2"]
}
```

## 5) Routing policy (recommended)

- If query contains proof/theorem language -> prioritize `GaussAgent`.
- If query contains intuition/analogy language -> prioritize `FeynmanAgent`.
- If query contains mechanics dynamics -> prioritize `NewtonAgent`.
- If quantum keywords -> use `GriffithsAgent` + `SakuraiAgent` and require manager arbitration.

## 6) Safety gate

If confidence < 0.55 or experts disagree on core equation, manager must call:

```json
{
  "type": "tool_call",
  "name": "request_clarification",
  "arguments": {
    "to_agent": "GaussAgent",
    "question": "Re-check the disputed derivation with explicit assumptions."
  }
}
```
