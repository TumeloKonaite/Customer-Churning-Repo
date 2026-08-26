# Production churn monitoring contract

| Field | Value |
| --- | --- |
| Contract version | `1.0.0` |
| Status | **PROPOSED — NOT APPROVED FOR PRODUCTION METRICS** |
| Owner | Monitoring Product Owner (name pending) |
| Required approvers | Business Churn Owner; Customer Data Owner; Security Officer; Privacy Officer |
| Approval date | Pending |
| Effective date | Pending; must be after all approvals |
| Implementation | `src/monitoring/shared/identity.py` |
| Change history | 1.0.0 (2026-08-20): initial proposed contract |

This document is the single versioned contract bundle for churn labels, horizons,
attribution, identity, privacy, retention, access, segmentation, and simulations.
“Must” and “must not” are normative. Values marked **Decision required** are safe
defaults, not assertions about an unknown source system.

## Activation and approval gate

No accuracy, precision, recall, calibration, drift-by-outcome, or other
label-dependent production metric is authoritative while this contract is not
`APPROVED`. A monitoring job must fail closed unless its configured contract
version has:

1. status `APPROVED`;
2. named people and dated approval records for all four required approver roles;
3. an effective date no later than the prediction being reported; and
4. no unresolved blocking decision for that effective version.

Approval means agreement with the business definition and governance controls; a
code review or merge is not business, security, privacy, or data-owner approval.
Approvers complete the record at the end of this document. A production release
must preserve the approved artifact and its commit SHA.

## 1. Churn-label contract

### Authoritative event

The v1 qualifying event is `CUSTOMER_RELATIONSHIP_TERMINATED`, sourced from the
authoritative Customer Master lifecycle stream. Its business timestamp is the
source event's `effective_at`, not its creation, ingestion, or processing time.
The source must provide `event_id`, `revision`, `effective_at`, tenant-scoped
customer identity, and retraction/correction state.

**Decision required (D-001):** the Customer Data Owner must confirm the exact
system, stream/table, event code, and meaning. Until then, the event name above is
a proposed canonical name and production labels are disabled.

### Semantics

- Churn is terminal and irreversible in v1. Reactivation is a new relationship,
  with a new stable customer identity. If the business treats reactivation as a
  reversal, v1 must be revised before activation.
- Only the event type above is qualifying. Additional churn types require a minor
  contract version, mapping to a canonical type, tests, and approval.
- The MVP label is binary: `1 = churned`, `0 = not churned`. `pending` is a label
  state, never coerced to `0`.
- An event qualifies for a prediction exactly when:

  ```text
  customer_token matches
  AND event_type is qualifying
  AND prediction_timestamp < effective_at <= horizon_end
  ```

- Duplicate delivery is idempotent on `(outcome_source, event_id, revision)`.
  Identical duplicates have no effect. Conflicting payloads at the same revision
  are quarantined.
- A higher revision supersedes a lower revision. A retraction removes the event
  from label calculation. Changed or retracted events rematerialize affected
  labels and mark already published reports as superseded; history remains
  auditable.
- Late events use `effective_at` for the window test. Processing time never moves
  an event into or out of a prediction window.

## 2. Prediction horizon and cohort maturity

| Rule | v1 value |
| --- | --- |
| Horizon duration | 90 calendar days (**Decision required D-002**) |
| Start | Persisted `prediction_timestamp` |
| End | `prediction_timestamp + 90 days` |
| Interval | Start exclusive, end inclusive: `(prediction_timestamp, horizon_end]` |
| Storage/comparison timezone | UTC only, timezone-aware ISO 8601 |
| Grace period | 7 calendar days after `horizon_end` (**Decision required D-003**) |
| Cohort maturity | `horizon_end + grace_period` |

An outcome immediately before or exactly at the prediction timestamp does not
label that prediction positive. An outcome exactly at the horizon end does. An
outcome immediately after the horizon end does not.

A qualifying positive may be materialized as soon as it is received. A negative
must not be materialized while `observed_at < horizon_end + grace_period`; the
state remains `pending`. A negative is allowed at or after the maturity instant.

Events arriving during the grace period are processed normally. A valid event
arriving after maturity still corrects the historical label; it also emits a
late-data quality signal and supersedes affected official report versions.
Producers must be monitored against the proposed seven-day service objective.

## 3. Prediction-to-outcome attribution

Every materialized label contains one exact, persisted `prediction_id`. Reports
join labels to predictions only by `prediction_id`; a customer-level “latest
prediction” join is prohibited.

- A `prediction_id` is an opaque globally unique ID assigned before the response
  is returned and persisted with its timestamp, horizon, customer token, model
  version, and contract version.
- Callers provide an idempotency key (`tenant + business_prediction_key`). A retry
  with the same key and identical request returns the original prediction. A
  changed request with the same key is rejected. Repeated intentional predictions
  use new keys and new IDs.
- A qualifying outcome is deterministically fanned out to **every** persisted
  prediction for the same token whose window contains the event. Each resulting
  label independently references its exact `prediction_id`. Sorting is by
  `(prediction_timestamp, prediction_id)` for repeatability.
- Multiple predictions and overlapping windows are allowed. There is no winner,
  latest-only rule, or destructive reassignment.
- A prediction at or after a previously known terminal churn event is rejected.
  If the earlier event arrives later, the prediction is quarantined and excluded
  from official metrics rather than labelled as a valid forecast.
- Duplicate outcome events do not create duplicate labels. Corrections rematerialize
  every affected prediction while retaining label revision history.

The current public prediction response does not expose `prediction_id` or accept a
trusted tokenized identity. The application persists privacy-safe prediction
events with `monitoring_eligible=false`; these rows must not be attributed to
outcomes. Trusted identity, horizon, and idempotency integration remain separate
prerequisites for production performance monitoring.

## 4. Customer identity and HMAC tokenization

The proposed authoritative identifier is the non-recycled, tenant-scoped
`customer_id` issued by Customer Master, not an account number, email address, or
dataset row number. **Decision required (D-004):** the Customer Data Owner must
confirm the system, stability and non-recycling guarantees, merge semantics, and
deletion signal.

Raw customer IDs must not be stored in the monitoring system. Tokenization is:

| Item | v1 rule |
| --- | --- |
| Algorithm | HMAC-SHA-256 with at least 256 bits of random key material |
| Canonical bytes | UTF-8 `v1\|len(environment):environment\|len(tenant_id):tenant_id\|len(customer_id):customer_id` |
| Output | `hmac-sha256:{key_id}:{lowercase_hex_digest}` |
| Namespace | Environment and tenant are mandatory |
| Secret storage | Managed KMS/HSM-backed secret store; never source, image, database, report, or log |
| Secret access | Tokenization workload identity and Security break-glass role only |
| Key ID | Non-secret immutable ID stored with each token |

The canonicalizer uses UTF-8 byte lengths, so delimiters within identifiers cannot
cause collisions. Input strings are not trimmed, case-folded, or Unicode-normalized;
the source system must provide its canonical value.

Keys rotate at least annually and immediately after suspected compromise. New
records use the active key. Historical records are not re-tokenized. Retired keys
remain decrypt-free HMAC material in the managed secret store only as long as
needed to match outcomes to still-retained prediction windows. During that period,
the tokenization service may generate tokens for the active and applicable retired
key IDs; monitoring storage never receives the keys. After customer merges, the
source supplies an approved alias-to-canonical mapping to the tokenization service.
Deletion removes customer-level tokens under the retention/deletion rules; it does
not reveal the raw ID to monitoring operators.

Raw identifiers, tokenization inputs or secrets, sensitive payloads, and raw or
transformed feature vectors must not enter application logs. Monitoring logging
uses a fixed message and an allow-list of structured fields.

## 5. Data classification and minimization

Monitoring stores only the following approved fields. A new field requires a
classification and contract change before collection.

| Field/data | Classification | Store? | Purpose and restriction |
| --- | --- | --- | --- |
| `prediction_id`, idempotency-key digest | Required operational data | Yes | Exact attribution/deduplication; never raw idempotency key |
| `prediction_timestamp`, `horizon_end`, `contract_version` | Required operational data | Yes | Window reproduction |
| model name/version/schema version | Required operational data | Yes | Reproducibility |
| `customer_token`, `key_id` | Quasi-identifier | Yes | Restricted customer correlation |
| raw customer/tenant/account identifiers | Personal information | No | Tokenize before boundary |
| `predicted_label`, `p_churn` | Monitoring feature | Yes | Performance/calibration; customer-level access restricted |
| raw model features | Sensitive information | No | Not required for v1 outcome metrics |
| transformed feature vectors/embeddings | Sensitive information | No | Reconstruction/inference risk |
| approved coarse segment attributes | Quasi-identifier | Yes | Only allow-listed values; protected by k threshold |
| free-text/request body/full headers/IP/user-agent | Sensitive information | No | Not needed; use allow-listed metadata only |
| request correlation ID, service version, environment | Required operational data | Yes | Troubleshooting; correlation ID must not encode identity |
| outcome ID/type/source/effective time/revision/retraction | Required operational data | Yes | Label auditability |
| `is_simulated`, generator/scenario version | Required operational data | Yes | Source separation |
| emails, phone numbers, names, addresses, credentials | Prohibited from storage | No | No monitoring purpose |

No protected or sensitive attribute may become a segment merely because it is
available to the model. Segment allow-list changes require Privacy approval.

## 6. Retention schedule

Retention starts at record creation unless noted. Shorter legal or contractual
limits prevail.

| Data | Maximum retention | End-of-life action |
| --- | --- | --- |
| Prediction events | 400 days | Delete customer-level record |
| Outcome events | 400 days | Delete customer-level record |
| Prediction labels and revisions | 400 days | Delete customer-level record |
| Customer-level monitoring datasets | 400 days | Delete and invalidate extracts |
| Aggregate Evidently/monitoring reports | 24 months | Delete report and cache |
| Extraction/job metadata | 90 days | Delete; retain only non-identifying audit summary |
| Operational logs | 30 days | Delete from primary and log archive |
| Simulated outcomes and reports | 30 days | Delete by scenario/run ID |

**Decision required (D-005):** Privacy, Security, and the Data Owner must approve
these periods against law, contract, incident-response, and business needs.

- Verified customer deletion triggers deletion of customer-token-level predictions,
  outcomes, labels, and extracts within 30 days. The identity service supplies the
  applicable key IDs/tokens; raw identity is not copied into monitoring.
- Immutable backups expire on their normal cycle within 90 days. Deleted data is
  not restored into live service; restore procedures replay deletion tombstones.
- A documented legal hold suspends only scoped destruction, records authority and
  expiry, restricts access, and resumes deletion when released.
- Aggregate reports may outlive customer-level data only if they contain no tokens,
  satisfy the segment policy, cannot be differenced to recover small cohorts, and
  are no longer traceable to a customer. Otherwise they share the 400-day limit.
- “Anonymization” must be validated as irreversible; tokenization alone is not
  anonymization.

## 7. Access-control matrix

`A` = allowed as a normal duty, `B` = time-limited break glass with review, `—` =
denied. Service identities and people use separate credentials. Production
application, migration, monitoring job, and report-reader credentials are distinct.

| Capability | Prediction app | Outcome ingestor | Label job | Monitoring analyst | Report viewer | Platform migration | Policy admin | Security break-glass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Write predictions | A | — | — | — | — | — | — | B |
| Ingest outcomes | — | A | — | — | — | — | — | B |
| Materialize labels | — | — | A | — | — | — | — | B |
| Read customer-level monitoring data | — | — | A | A | — | — | — | B |
| Run monitoring jobs | — | — | A | A | — | — | — | B |
| View aggregate reports | — | — | A | A | A | — | A | B |
| Change schema/migrate storage | — | — | — | — | — | A | — | B |
| Change monitoring policies/contracts | — | — | — | — | — | — | A | B |
| Change retention settings | — | — | — | — | — | — | A | B |
| Read tokenization secret | Tokenization service only | Tokenization service only | — | — | — | — | — | B |

Policy-admin changes require pull-request review plus the approver roles affected
by the change. Retention reductions need Privacy and Data Owner approval; increases
also need Security approval. All grants, job runs, customer-level reads, policy
changes, exports, and break-glass events are audited. Analysts cannot export
customer-level data to unmanaged tools.

## 8. Segment-suppression policy

The configurable default is `k = 20` distinct mature prediction IDs after all
filters. `k` must not be configured below 20 in production without a new Privacy-
approved contract version.

- For a segment with `n < k`, suppress the segment count and **all** statistics,
  including performance, probability, drift, calibration, fairness, and confidence
  interval values. Internal logs must not contain the suppressed count or metric.
- v1 does not combine suppressed segments into `Other`; doing so can reveal a
  suppressed value by subtraction. Any future `Other` bucket must itself have
  `n >= k` and pass a differencing review.
- Official comparisons use registered, non-overlapping calendar windows and a
  fixed segment allow-list. Arbitrary filters, adjacent/sliding windows, drill-down,
  and exports that permit differencing are disabled for report viewers.
- A privacy review is required before publishing intersecting segments or changing
  time-window granularity. Suppression is applied after intersections and filters,
  never only to marginal totals.

## 9. Simulated-outcome policy

Every outcome has both a non-empty `outcome_source` and explicit `is_simulated`.
A simulated event additionally requires `generator_version` and
`scenario_version`; a real event must not set either generator field.

- Simulated outcome ingestion is allowed only in local, test, development, and
  staging environments. Production ingestion rejects it at the boundary.
- Recommended source format is `simulation:{generator}:{scenario}`. Production
  source names are allow-listed independently after D-001 is resolved.
- A dataset or report must contain only real outcomes or only one declared
  simulation run. Mixing real and simulated outcomes is prohibited.
- Official reports accept only real outcomes. Simulation reports carry metadata
  `official=false`, `is_simulated=true`, generator/scenario versions and run ID,
  and display `SIMULATED — NOT PRODUCTION PERFORMANCE` in title and export.
- Simulation data is isolated by environment and storage prefix, deleted within
  30 days, and removable immediately by run ID. It must never be promoted or
  copied into production monitoring tables.

## 10. Versioning, changes, and approval record

Contract versions use semantic versioning. A breaking definition, boundary,
identity, or privacy change increments major; an additive event/field/role change
increments minor; clarification with no behavior change increments patch. Stored
predictions and labels retain the version effective when created; historical
reports are not silently recomputed under a new version.

Any change includes rationale, decision-log updates, executable examples, migration
and report-restatement impact, owner, all affected approvers, approval/effective
dates, and an immutable change history entry. Unresolved assumptions never become
defaults in production.

### Approval record

| Role | Named approver | Decision | Approval date | Evidence/reference |
| --- | --- | --- | --- | --- |
| Monitoring Product Owner | Pending | Pending | Pending | Pending |
| Business Churn Owner | Pending | Pending | Pending | Pending |
| Customer Data Owner | Pending | Pending | Pending | Pending |
| Security Officer | Pending | Pending | Pending | Pending |
| Privacy Officer | Pending | Pending | Pending | Pending |

The status at the top may change to `APPROVED` only when every row is completed
and all blocking decisions below are resolved.

## Decision log

| ID | Blocking? | Decision/assumption | Proposed v1 value | Owner | Status |
| --- | ---: | --- | --- | --- | --- |
| D-001 | Yes | Exact qualifying business event and authoritative source | Customer Master `CUSTOMER_RELATIONSHIP_TERMINATED` / `effective_at`; terminal | Business Churn Owner + Data Owner | Open |
| D-002 | Yes | Prediction horizon | 90 calendar days | Business Churn Owner | Open |
| D-003 | Yes | Late-arrival grace period and source SLO | 7 calendar days | Data Owner | Open |
| D-004 | Yes | Stable identity, non-recycling, merge/deletion behavior | Tenant-scoped Customer Master ID | Data Owner + Privacy | Open |
| D-005 | Yes | Retention/deletion/backup periods | Schedule in section 6 | Privacy + Security + Data Owner | Open |
| D-006 | Yes | Minimum segment size and allowed segment attributes | k=20; allow-list not yet supplied | Privacy + Business Churn Owner | Open |
| D-007 | Yes | Production event source allow-list and environment controls | Real source pending; simulations forbidden in production | Security + Data Owner | Open |
| D-008 | No | Attribution across overlapping windows | Fan out to every eligible persisted prediction ID | Monitoring Product Owner | Proposed |
| D-009 | No | Historical token handling after key rotation | No retokenization; retain applicable keys for matching retained windows | Security | Proposed |

## Executable examples and traceability

`src/monitoring/shared/identity.py` is the executable reference for the unambiguous v1
rules. `tests/test_monitoring_contracts.py` covers:

- immediately before/exactly at prediction time and exactly at/immediately after
  horizon end;
- rejection of premature negatives and allowance at cohort maturity;
- multiple and overlapping prediction windows with exact IDs;
- idempotent duplicates and deterministic correction/retraction behavior;
- deterministic HMACs and namespace separation;
- rejection of raw identifiers in monitoring logs;
- suppression below k; and
- rejection of simulated/real mixtures and simulations in official reports.

Executable examples do not replace approval. The fail-closed activation gate at
the beginning remains authoritative.
