# Cycle 2C Phase A preflight

Protocol preparation and software preflight only. **NO LONG SCIENTIFIC CYCLE 2C RUN HAS BEEN EXECUTED.** No 150-task / 9,000,000-update run or static-sparse gate result exists.

The real-MNIST preflight processed the first 8,192 examples of task 0 for each of the four sparse methods. It verified the real dataset source, exact task-sequence hash, identical initialization, identical learner state through update 8,191, the first intervention at update 8,192, conserved edge budgets, finite state, and checkpoint roundtrips. These checks are software validation, not scientific evidence.

The methods use 99,533 logical active parameters including biases. `protocol.json` contains the frozen choices; `protocol_lock.json` records SHA256 hashes for 23 scientifically relevant sources and configs. `test_summary.json` records the complete unit suite and legacy synthetic smoke.

Independently review this protocol-ready branch before running `./scripts/run_cycle2c.sh`. The runner performs the 150-task Static Sparse gate first and only then launches dynamic comparisons if every preregistered condition passes.
