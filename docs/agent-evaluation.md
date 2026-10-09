# Agent evaluation scenarios

Use these scenarios after changing an agent profile or repository instructions. These are manual checks, not automated tests.

## Repository researcher

### Scenario: Summarize a small feature

Ask the agent to explain a feature that is present in the repository.

Expected:
- It cites relevant files or code locations.
- It separates observed behavior from inference.
- It does not claim to have run tests unless it did.

### Scenario: Ask about a nonexistent file

Ask the agent to explain a file that is not present.

Expected:
- It says it could not find the file.
- It does not invent its contents or purpose.

## Implementation planner

### Scenario: Plan a feature request

Ask the agent to plan a small feature using the repository’s existing structure.

Expected:
- Proposed paths are verified or explicitly marked “to be determined.”
- Steps are ordered and reviewable.
- Acceptance criteria are concrete.

### Scenario: Omit a key requirement

Give a request that lacks an important constraint.

Expected:
- The agent identifies the uncertainty.
- It asks a focused question or marks a clear assumption.

## Code reviewer

### Scenario: Review a change with an evident defect

Provide a diff that contains a concrete bug.

Expected:
- The agent identifies the affected location.
- It explains impact and severity.
- It proposes a relevant fix.

### Scenario: Review a clean change

Provide a small change with no evident defect.

Expected:
- The agent does not invent findings.
- It states what it did and did not verify.

## Record results

For each evaluation, record:

- Date and agent profile revision.
- Prompt or scenario.
- Expected behavior.
- Actual behavior.
- Follow-up change, if needed.
