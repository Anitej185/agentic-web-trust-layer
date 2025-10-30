"""
Pydantic models for AgentCert MVP
"""
from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class AgentTask(BaseModel):
    """Task request for the agent"""
    prompt: str = Field(..., description="Financial scenario or question")
    role: str = "You are a responsible financial advisor providing general educational information."


class AgentResponse(BaseModel):
    """Agent's response to a task"""
    prompt: str
    response: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Evaluation(BaseModel):
    """Evaluation scores for agent response"""
    prompt: str
    response: str
    accuracy: int = Field(..., ge=0, le=100, description="Accuracy score 0-100")
    clarity: int = Field(..., ge=0, le=100, description="Clarity score 0-100")
    compliance: int = Field(..., ge=0, le=100, description="Compliance score 0-100")
    feedback: str = Field(..., description="Feedback summary")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def average_score(self) -> float:
        """Calculate average score"""
        return (self.accuracy + self.clarity + self.compliance) / 3


class Certification(BaseModel):
    """Certification JSON issued to agent"""
    agent_name: str = "Financial Advisor Agent"
    certified: bool = False
    scores: Dict[str, float] = Field(default_factory=dict)
    issued_at: Optional[datetime] = None
    verifier: str = "AgentCert MVP"
    attempts: int = 0
    passed_threshold: float = 85.0


class EvaluationHistory(BaseModel):
    """Collection of evaluations"""
    evaluations: List[Evaluation] = Field(default_factory=list)
    
    @property
    def average_accuracy(self) -> float:
        if not self.evaluations:
            return 0.0
        return sum(e.accuracy for e in self.evaluations) / len(self.evaluations)
    
    @property
    def average_clarity(self) -> float:
        if not self.evaluations:
            return 0.0
        return sum(e.clarity for e in self.evaluations) / len(self.evaluations)
    
    @property
    def average_compliance(self) -> float:
        if not self.evaluations:
            return 0.0
        return sum(e.compliance for e in self.evaluations) / len(self.evaluations)
    
    @property
    def overall_average(self) -> float:
        if not self.evaluations:
            return 0.0
        return sum(e.average_score for e in self.evaluations) / len(self.evaluations)

