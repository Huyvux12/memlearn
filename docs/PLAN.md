# Execution plan — 2026-10-17
Spec: docs/IMPLEMENTATION_SPEC.md
1. Write failing generator/oracle/parser/statistics tests. Implement deterministic data and metrics. Interfaces: Bundle->Request; Request gold never passed to model.
2. Write failing mathematical and real tiny-Qwen tests. Implement version-scoped eager adapter with causal mask/cache/GQA and freeze-safe training modules. Interfaces: attach(model, method, layers, rank), adapter_state.
3. Write failing run/artifact/resume tests. Implement training, inference timing, balanced schedules, atomic receipts, report/audit and CLI. Interfaces: immutable config fingerprint, run directory; no partial-as-full.
4. Package standalone notebook, requirements and Vietnamese guide. Test exact source payload roundtrip and compile cells. Run software smoke with random tiny Qwen only.
5. Fresh whole-project review, one regression fix pass, full suite, save notebook/source zip/guide.

Global constraints: isolated new project, no edits to agent code, no API calls, no fabricated neural results, same backbone/data/budget across trained variants, no evaluator threshold tuning.
