# ClauseLens Manual Test Questions

This document contains manual test questions that are not included in
`docs/notes.md`. Use them to test topic routing, retrieval quality, grounded
answers, conversation context, citations, and abstention.

## Anti-Assignment

Expected clause type: `Anti-Assignment`

```text
What happens if a party attempts an assignment without consent?
Does the assignment restriction apply to transfers by operation of law?
Is assignment permitted to a wholly owned subsidiary?
Can approval for a sublicense be withheld at a party's sole discretion?
May responsibilities under the agreement be delegated to another company?
```

Checks:

- The answer distinguishes assignment, transfer, delegation, and sublicensing.
- An attempted unauthorized assignment is described as void only when the
  retrieved evidence says so.
- Exceptions are attributed to their specific source agreement.

## Cap On Liability

Expected clause type: `Cap On Liability`

```text
Are lost profits and anticipated savings recoverable?
Does the liability limitation exclude punitive and consequential damages?
Are intellectual-property infringement claims exempt from the liability limitation?
Can unpaid royalties or milestones qualify as direct damages?
Which categories of loss are excluded by the agreement?
```

Checks:

- Excluded damages and exceptions are not combined across agreements.
- The answer does not describe every liability limitation as a monetary cap.
- Direct and consequential damages are distinguished where the evidence does
  so.

## License Grant

Expected clause type: `License Grant`

```text
Is the granted license exclusive or non-exclusive?
Is the license perpetual and irrevocable?
Does the license permit sublicensing?
May the licensee transfer the license?
Is the license worldwide or restricted to a particular territory?
What activities, such as making, selling, importing, or displaying, are permitted?
Does the license permit access to paid premium features?
How long does the license remain effective?
```

Checks:

- Sublicensing is not inferred from assignment or transfer language.
- Transferability is not inferred from sublicensing language.
- Scope, duration, territory, and exclusivity are supported by citations.
- Different license grants are presented as separate source-specific examples.

## Audit Rights

Expected clause type: `Audit Rights`

```text
How often may a party conduct an audit?
How much advance notice is required before an audit?
Are there periods when an audit cannot be conducted?
Who pays the audit costs if a payment deficiency exceeds three percent?
Can the auditor copy or take extracts from the records?
Does accepting a payment prevent a party from later disputing its accuracy?
Where must an audit be conducted?
Who is permitted to perform the audit?
```

Checks:

- Notice periods, frequency limits, and cost-shifting thresholds are quoted
  accurately.
- Conditions from different agreements are not merged.
- The answer says when a requested detail is absent from the retrieved
  evidence.

## Termination For Convenience

Expected clause type: `Termination For Convenience`

```text
Who specifically has the right to terminate without cause?
Can termination occur immediately, or is advance notice required?
Does the termination right become available only after a particular anniversary?
Can an individual service be terminated without terminating the entire agreement?
What costs must the terminating party pay after ending a service?
Does the agreement automatically terminate when another agreement ends?
Is the termination right available to both parties or only one party?
```

Checks:

- The answer identifies the party holding the termination right.
- Notice periods and timing conditions remain source-specific.
- Service termination is distinguished from termination of the entire
  agreement.

## Conversation Context

Run each group as one continuous conversation.

### License follow-ups

```text
Does the license permit sublicensing?
Is it also transferable?
How long does it remain effective?
```

Expected behavior:

- All three questions resolve to `License Grant`.
- The final standalone query retains the license topic.
- Transferability and sublicensing are answered as separate rights.

### Audit follow-ups

```text
How often can records be audited?
How much notice is needed?
Who pays if the audit finds a large underpayment?
```

Expected behavior:

- All three questions resolve to `Audit Rights`.
- Follow-up queries preserve the audit topic.
- The cost-shifting threshold is included only when supported by evidence.

### Unsupported topic followed by a supported topic

```text
Does the contract require arbitration?
Does the license permit sublicensing?
Is it also transferable?
How long does it remain effective?
```

Expected behavior:

- The arbitration question abstains without retrieval.
- The license question starts a new topic and its standalone query does not
  contain the arbitration question.
- The remaining follow-ups preserve `License Grant`.

## Unsupported Topics

Expected behavior: abstain with no retrieved evidence.

```text
Is either party required to indemnify the other?
Does the contract require arbitration?
Does the agreement contain a confidentiality obligation?
What events qualify as force majeure?
Is there a non-compete restriction?
What law governs the agreement?
Does the agreement renew automatically?
What remedies are available after a breach?
```

## General Acceptance Checks

For every supported question:

- The resolved clause type matches the legal topic.
- The answer is supported by the retrieved text.
- Every citation refers to returned evidence.
- Different agreements are not merged into a synthetic contract rule.
- Missing details are acknowledged rather than invented.
- Answers remain complete and concise.

For every unsupported question:

- The response abstains.
- No evidence is returned.
- Retrieval and answer generation are skipped.
