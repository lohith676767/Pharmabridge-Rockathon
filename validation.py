"""
validation.py
The deterministic Python layer between Agent 1 and Agent 2.

This is the "innovation" piece: it is NOT an LLM. It is plain rule-based
logic that decides whether Agent 2 is even allowed to run. This is what
lets you say on stage: "Agent 2 doesn't trust Agent 1 - it knows when
Agent 1 should not yet be trusted."
"""

from datetime import datetime, timezone
from schemas import ProcessKnowledgePackage, ValidationIssue, ValidationResult

REQUIRED_FIELDS = ["parameter", "target_value", "criticality", "scale_sensitivity", "evidence"]
STALE_DAYS_THRESHOLD = 365 * 2  # anything older than ~2 years is treated as stale for demo purposes


def validate_package(pkg: ProcessKnowledgePackage) -> ValidationResult:
    blocking = []
    warnings = []

    # 1. Critical data missing
    for field in REQUIRED_FIELDS:
        value = getattr(pkg, field, None)
        if value in (None, "", []):
            blocking.append(ValidationIssue(
                code="MISSING_CRITICAL_FIELD",
                severity="blocking",
                message=f"Required field '{field}' is missing. Agent 2 cannot safely determine controls without it.",
            ))

    # 2. Two values conflict: target outside its own validated range
    if pkg.target_value is not None and pkg.validated_range_low is not None and pkg.validated_range_high is not None:
        if not (pkg.validated_range_low <= pkg.target_value <= pkg.validated_range_high):
            blocking.append(ValidationIssue(
                code="TARGET_OUTSIDE_RANGE",
                severity="blocking",
                message=(f"Target value {pkg.target_value} falls outside the validated range "
                          f"{pkg.validated_range_low}-{pkg.validated_range_high}. Explicit contradiction."),
            ))

    # 2b. Target exceeds a hard safety/risk cap from risk assessment
    if pkg.target_value is not None and pkg.risk_cap is not None:
        if pkg.target_value > pkg.risk_cap:
            blocking.append(ValidationIssue(
                code="TARGET_EXCEEDS_RISK_CAP",
                severity="blocking",
                message=(f"Target value {pkg.target_value} exceeds the risk-assessment safety cap "
                          f"of {pkg.risk_cap}. PM target and risk assessment disagree."),
            ))

    # 3. "Validated" at pilot scale only -> flag as scale gap (non-blocking, but must be surfaced)
    if pkg.evidence_scale and pkg.evidence_scale.lower() == "pilot":
        warnings.append(ValidationIssue(
            code="SCALE_GAP_UNVERIFIED",
            severity="warning",
            message="Evidence is pilot-scale only. Commercial-scale behaviour is unverified and must be flagged as an open risk.",
        ))

    # 4. Hidden dependency -> must be traced through, not silently dropped
    if pkg.dependencies:
        warnings.append(ValidationIssue(
            code="DEPENDENCY_LINKED",
            severity="warning",
            message=f"Parameter change may affect: {', '.join(pkg.dependencies)}. Must be traced into risk assessment.",
        ))

    # 5. Outdated information -> stale data should not be treated as current
    if pkg.last_updated:
        try:
            updated = datetime.fromisoformat(pkg.last_updated).replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - updated).days
            if age_days > STALE_DAYS_THRESHOLD:
                warnings.append(ValidationIssue(
                    code="STALE_DATA",
                    severity="warning",
                    message=f"Source data is {age_days} days old. Previous validation marked STALE, not current.",
                ))
        except ValueError:
            warnings.append(ValidationIssue(
                code="UNPARSEABLE_DATE",
                severity="warning",
                message="last_updated could not be parsed; treat validation currency as unknown.",
            ))

    # 6. Perfect-looking but incomplete: all numbers present but no safety rationale
    numbers_present = pkg.target_value is not None and pkg.validated_range_low is not None
    if numbers_present and not pkg.safety_rationale and not pkg.uncertainty:
        warnings.append(ValidationIssue(
            code="MISSING_SAFETY_RATIONALE",
            severity="warning",
            message="Numeric parameters are complete, but the WHY (safety rationale / uncertainty) is missing. "
                     "A perfect-looking handoff can still be an unsafe one.",
        ))

    return ValidationResult(passed=(len(blocking) == 0), blocking_issues=blocking, warnings=warnings)
