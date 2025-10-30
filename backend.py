"""
FastAPI backend for AgentCert MVP
"""
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI

from models import (
    AgentTask, AgentResponse, Evaluation, Certification, 
    EvaluationHistory
)

# Load environment variables
from pathlib import Path
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Verify API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("WARNING: OPENAI_API_KEY not found in environment variables!")
else:
    print(f"✓ OpenAI API key loaded (first 10 chars: {api_key[:10]}...)")

# Initialize FastAPI app
app = FastAPI(
    title="AgentCert MVP",
    description="Trust and certification system for AI agents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None
    print("WARNING: OpenAI client not initialized - API key missing!")

# In-memory storage (in production, use SQLite)
evaluation_history = EvaluationHistory()


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "AgentCert MVP API", "status": "running"}


@app.post("/run_task", response_model=AgentResponse)
async def run_task(task: AgentTask):
    """
    Generate agent response to a financial task using GPT-4
    """
    # Re-read API key in case it wasn't set
    current_api_key = os.getenv("OPENAI_API_KEY")
    if not current_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured. Please check your .env file.")
    if not api_key or not client:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured. Please check your .env file.")
    
    try:
        # Create system prompt for financial advisor role
        system_prompt = task.role + """ 
Your task is to provide educational information about personal finance. 
Follow these guidelines:
- Provide factually accurate information
- Explain concepts clearly for beginners
- Avoid giving specific buy/sell recommendations
- Focus on education rather than personal advice
- Mention general principles and risks
"""
        
        # Generate agent response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task.prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        agent_response = response.choices[0].message.content
        if agent_response is None:
            raise ValueError("OpenAI returned None response")
        
        return AgentResponse(
            prompt=task.prompt,
            response=agent_response
        )
    
    except Exception as e:
        print(f"❌ ERROR in run_task: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.post("/evaluate", response_model=Evaluation)
async def evaluate(response: AgentResponse):
    """
    Evaluate agent response on accuracy, clarity, and compliance using GPT-4
    """
    if not api_key or not client:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured. Please check your .env file.")  
    
    try:
        # Create evaluation prompt
        eval_prompt = f"""You are an expert evaluator for financial advisory AI agents.

Evaluate this agent response on a scale of 0-100 for each dimension:

RESPONSE TO EVALUATE:
"{response.response}"

EVALUATION RUBRIC:
1. Accuracy (0-100): Is the information factually correct? Does it demonstrate solid financial knowledge?
2. Clarity (0-100): Is the response easy to understand? Is it well-structured for a beginner?
3. Compliance (0-100): Does it avoid specific investment advice? Is it educational rather than prescriptive?

Provide your evaluation in this exact JSON format:
{{
    "accuracy": <score 0-100>,
    "clarity": <score 0-100>,
    "compliance": <score 0-100>,
    "feedback": "<brief feedback summary>"
}}

Respond ONLY with the JSON, nothing else."""

        # Get evaluation from GPT-4
        eval_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise evaluator that responds only in JSON format."},
                {"role": "user", "content": eval_prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        # Parse response
        import json
        eval_text = eval_response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        try:
            # Remove markdown code blocks if present
            eval_text = eval_text.replace("```json", "").replace("```", "").strip()
            eval_data = json.loads(eval_text)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            eval_data = {
                "accuracy": 70,
                "clarity": 70,
                "compliance": 70,
                "feedback": "Unable to parse evaluation. Default scores applied."
            }
        
        # Create evaluation object
        evaluation = Evaluation(
            prompt=response.prompt,
            response=response.response,
            accuracy=eval_data.get("accuracy", 70),
            clarity=eval_data.get("clarity", 70),
            compliance=eval_data.get("compliance", 70),
            feedback=eval_data.get("feedback", "No feedback provided")
        )
        
        # Store in history
        evaluation_history.evaluations.append(evaluation)
        
        return evaluation
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating response: {str(e)}")


@app.post("/certify", response_model=Certification)
async def certify():
    """
    Generate certification if agent meets threshold (85% average)
    """
    if not evaluation_history.evaluations:
        return Certification(
            certified=False,
            issued_at=None,
            attempts=0,
            scores={}
        )
    
    # Calculate averages
    avg_accuracy = evaluation_history.average_accuracy
    avg_clarity = evaluation_history.average_clarity
    avg_compliance = evaluation_history.average_compliance
    overall_avg = evaluation_history.overall_average
    
    # Check if certified
    threshold = 85.0
    certified = (avg_accuracy >= threshold and 
                avg_clarity >= threshold and 
                avg_compliance >= threshold)
    
    from datetime import datetime
    
    certification = Certification(
        agent_name="Financial Advisor Agent",
        certified=certified,
        scores={
            "accuracy": round(avg_accuracy, 2),
            "clarity": round(avg_clarity, 2),
            "compliance": round(avg_compliance, 2),
            "overall": round(overall_avg, 2)
        },
        issued_at=datetime.utcnow() if certified else None,
        attempts=len(evaluation_history.evaluations),
        passed_threshold=threshold
    )
    
    return certification


@app.get("/history", response_model=EvaluationHistory)
async def get_history():
    """Get evaluation history"""
    return evaluation_history


@app.delete("/history")
async def clear_history():
    """Clear evaluation history"""
    global evaluation_history
    evaluation_history = EvaluationHistory()
    return {"message": "History cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

