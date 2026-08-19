"""
schemas.py
Strict data contracts for the R&D -> Manufacturing handoff.

These schemas ARE the "context-preserving" mechanism: Agent 1 cannot
hand off anything that doesn't fit this shape, and Agent 2 cannot
receive anything that doesn't fit this shape. Nothing free-text
crosses the boundary between agents.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class ProcessKnowledgePackage(BaseModel):
    """Output of Agent 1 (Product Manager). Input contract for the validation layer."""

    parameter: str = Field(..., description="Name of the critical process parameter, e.g. Temperature")
    target_value: Optional[float] = Field(None, description="Target value, e.g. 52")
    unit: Optional[str] = Field(None, description="Unit, e.g. °C")
    validated_range_low: Optional[float] = None
    validated_range_high: Optional[float] = None
    criticality: Optional[str] = Field(None, description="High / Medium / Low")
    quality_impact: Optional[str] = None
    scale_sensitivity: Optional[str] = Field(None, description="High / Medium / Low")
    evidence: Optional[str] = Field(None, description="e.g. 'Pilot validation data'")
    evidence_scale: Optional[str] = Field(None, description="'pilot' or 'commercial'")
    uncertainty: Optional[str] = Field(None, description="Known unknowns, e.g. 'Scale-up unvalidated'")
    risk_cap: Optional[float] = Field(None, description="Hard safety ceiling if one exists, from risk assessment")
    dependencies: List[str] = Field(default_factory=list, description="Other parameters/quality attributes this affects")
    last_updated: Optional[str] = Field(None, description="ISO date the source data was last validated/updated")
    safety_rationale: Optional[str] = Field(None, description="WHY this range/limit exists, not just what it is")

    @field_validator("criticality", "scale_sensitivity")
    @classmethod
    def normalize_level(cls, v):
        if v is None:
            return v
        return v.strip().title()


class ValidationIssue(BaseModel):
    code: str
    severity: str  # "blocking" or "warning"
    message: str


class ValidationResult(BaseModel):
    passed: bool
    blocking_issues: List[ValidationIssue] = Field(default_factory=list)
    warnings: List[ValidationIssue] = Field(default_factory=list)


class ManufacturingDesign(BaseModel):
    """Output of Agent 2 (Solution Architect)."""

    control_instruction: str
    monitoring: str
    validation_requirement: str
    deviation_handling: str
    risk_assessment: str
    traceability: str
    open_risk_flags: List[str] = Field(default_factory=list)
