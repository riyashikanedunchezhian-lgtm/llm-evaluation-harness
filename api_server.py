"""FastAPI server for programmatic access to LLM evaluation harness."""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import sys
import json
from datetime import datetime
import uuid

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.harness import EvaluationHarness
from src.config import MODEL_CONFIGS

app = FastAPI(
    title="LLM Evaluation API",
    description="REST API for LLM evaluation with jury-style scoring",
    version="2.0.0"
)

# Store running evaluations
running_evaluations: Dict[str, Dict] = {}

class EvaluationRequest(BaseModel):
    """Request model for starting an evaluation."""
    models: List[str] = Field(
        ...,
        description="List of model IDs to evaluate",
        example=["claude-3-5-sonnet-20241022", "gpt-4o"]
    )
    categories: Optional[List[str]] = Field(
        None,
        description="Categories to evaluate (default: all)",
        example=["factual_qa", "reasoning"]
    )
    bias_check: bool = Field(
        False,
        description="Enable position bias checking"
    )
    sequential: bool = Field(
        False,
        description="Use sequential judge execution instead of parallel"
    )
    test_set_path: str = Field(
        "prompts/test_set.json",
        description="Path to test set JSON file"
    )

class EvaluationStatus(BaseModel):
    """Response model for evaluation status."""
    evaluation_id: str
    status: str
    progress: float
    current_task: str
    total_tests: int
    completed_tests: int
    started_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None

class EvaluationResults(BaseModel):
    """Response model for evaluation results."""
    evaluation_id: str
    status: str
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    export_urls: Dict[str, str]

def progress_callback(progress_info: Dict[str, Any]):
    """Progress callback for API evaluations."""
    evaluation_id = progress_info.get("evaluation_id")
    if evaluation_id and evaluation_id in running_evaluations:
        running_evaluations[evaluation_id]["progress_info"] = progress_info
        
        if "status" in progress_info:
            if progress_info["status"] == "complete":
                running_evaluations[evaluation_id]["status"] = "completed"
                running_evaluations[evaluation_id]["completed_at"] = datetime.now().isoformat()
            elif progress_info["status"] == "error":
                running_evaluations[evaluation_id]["status"] = "error"
                running_evaluations[evaluation_id]["error"] = progress_info.get("error", "Unknown error")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LLM Evaluation API",
        "version": "2.0.0",
        "description": "REST API for LLM evaluation with jury-style scoring",
        "endpoints": {
            "models": "/api/v1/models",
            "start_evaluation": "/api/v1/evaluations",
            "get_status": "/api/v1/evaluations/{evaluation_id}",
            "get_results": "/api/v1/evaluations/{evaluation_id}/results",
            "cancel_evaluation": "/api/v1/evaluations/{evaluation_id}/cancel"
        }
    }

@app.get("/api/v1/models")
async def get_models():
    """Get available models."""
    return {
        "models": [
            {
                "id": model_id,
                "name": config.name,
                "provider": config.provider,
                "input_price_per_1k": config.input_price_per_1k,
                "output_price_per_1k": config.output_price_per_1k,
                "max_tokens": config.max_tokens
            }
            for model_id, config in MODEL_CONFIGS.items()
        ]
    }

@app.post("/api/v1/evaluations", response_model=EvaluationStatus)
async def start_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """Start a new evaluation."""
    # Validate models
    for model_id in request.models:
        if model_id not in MODEL_CONFIGS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model '{model_id}'. Available models: {list(MODEL_CONFIGS.keys())}"
            )
    
    # Generate evaluation ID
    evaluation_id = str(uuid.uuid4())
    
    # Initialize evaluation state
    running_evaluations[evaluation_id] = {
        "status": "running",
        "progress": 0.0,
        "current_task": "Initializing",
        "total_tests": 0,
        "completed_tests": 0,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "request": request.dict(),
        "harness": None,
        "results": None
    }
    
    # Define background task
    def run_evaluation_task():
        try:
            # Initialize harness
            harness = EvaluationHarness(
                model_ids=request.models,
                test_set_path=request.test_set_path,
                parallel_judge=not request.sequential,
                progress_callback=lambda info: progress_callback({**info, "evaluation_id": evaluation_id})
            )
            
            running_evaluations[evaluation_id]["harness"] = harness
            
            # Run evaluation
            results = harness.run_evaluation(
                categories=request.categories,
                enable_bias_check=request.bias_check
            )
            
            # Save results
            output_path = f"data/results_{evaluation_id}.json"
            harness.save_results(output_path)
            
            # Store results
            running_evaluations[evaluation_id]["results"] = results
            running_evaluations[evaluation_id]["output_path"] = output_path
            running_evaluations[evaluation_id]["status"] = "completed"
            running_evaluations[evaluation_id]["completed_at"] = datetime.now().isoformat()
            
        except Exception as e:
            running_evaluations[evaluation_id]["status"] = "error"
            running_evaluations[evaluation_id]["error"] = str(e)
            running_evaluations[evaluation_id]["completed_at"] = datetime.now().isoformat()
    
    # Add background task
    background_tasks.add_task(run_evaluation_task)
    
    return EvaluationStatus(
        evaluation_id=evaluation_id,
        status="running",
        progress=0.0,
        current_task="Initializing",
        total_tests=0,
        completed_tests=0,
        started_at=running_evaluations[evaluation_id]["started_at"]
    )

@app.get("/api/v1/evaluations/{evaluation_id}", response_model=EvaluationStatus)
async def get_evaluation_status(evaluation_id: str):
    """Get status of an evaluation."""
    if evaluation_id not in running_evaluations:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    eval_state = running_evaluations[evaluation_id]
    
    # Update progress from progress callback
    if "progress_info" in eval_state:
        progress_info = eval_state["progress_info"]
        if "progress_percent" in progress_info:
            eval_state["progress"] = progress_info["progress_percent"] / 100
        if "current" in progress_info and "total" in progress_info:
            eval_state["completed_tests"] = progress_info["current"]
            eval_state["total_tests"] = progress_info["total"]
        if "test_id" in progress_info:
            eval_state["current_task"] = f"Evaluating {progress_info['test_id']}"
    
    return EvaluationStatus(
        evaluation_id=evaluation_id,
        status=eval_state["status"],
        progress=eval_state.get("progress", 0.0),
        current_task=eval_state.get("current_task", "Unknown"),
        total_tests=eval_state.get("total_tests", 0),
        completed_tests=eval_state.get("completed_tests", 0),
        started_at=eval_state["started_at"],
        completed_at=eval_state.get("completed_at"),
        error=eval_state.get("error")
    )

@app.get("/api/v1/evaluations/{evaluation_id}/results")
async def get_evaluation_results(evaluation_id: str):
    """Get results of a completed evaluation."""
    if evaluation_id not in running_evaluations:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    eval_state = running_evaluations[evaluation_id]
    
    if eval_state["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Evaluation not completed. Current status: {eval_state['status']}"
        )
    
    if not eval_state.get("results"):
        raise HTTPException(status_code=404, detail="Results not available")
    
    harness = eval_state["harness"]
    summary = harness.get_aggregated_metrics()
    
    return EvaluationResults(
        evaluation_id=evaluation_id,
        status=eval_state["status"],
        results=eval_state["results"],
        summary=summary,
        export_urls={
            "json": f"/api/v1/evaluations/{evaluation_id}/export/json",
            "csv": f"/api/v1/evaluations/{evaluation_id}/export/csv"
        }
    )

@app.get("/api/v1/evaluations/{evaluation_id}/export/json")
async def export_results_json(evaluation_id: str):
    """Export results as JSON."""
    if evaluation_id not in running_evaluations:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    eval_state = running_evaluations[evaluation_id]
    
    if eval_state["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Evaluation not completed. Current status: {eval_state['status']}"
        )
    
    return JSONResponse(content=eval_state["results"])

@app.get("/api/v1/evaluations/{evaluation_id}/export/csv")
async def export_results_csv(evaluation_id: str):
    """Export results as CSV."""
    if evaluation_id not in running_evaluations:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    eval_state = running_evaluations[evaluation_id]
    
    if eval_state["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Evaluation not completed. Current status: {eval_state['status']}"
        )
    
    harness = eval_state["harness"]
    df = harness.get_summary_dataframe()
    
    from fastapi.responses import Response
    csv_content = df.to_csv(index=False)
    
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=evaluation_{evaluation_id}.csv"
        }
    )

@app.delete("/api/v1/evaluations/{evaluation_id}/cancel")
async def cancel_evaluation(evaluation_id: str):
    """Cancel a running evaluation."""
    if evaluation_id not in running_evaluations:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    eval_state = running_evaluations[evaluation_id]
    
    if eval_state["status"] == "completed":
        raise HTTPException(status_code=400, detail="Cannot cancel completed evaluation")
    
    if eval_state["status"] == "error":
        raise HTTPException(status_code=400, detail="Evaluation already failed")
    
    # Mark as cancelled
    eval_state["status"] = "cancelled"
    eval_state["completed_at"] = datetime.now().isoformat()
    
    return {"message": "Evaluation cancelled", "evaluation_id": evaluation_id}

@app.get("/api/v1/evaluations")
async def list_evaluations(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(10, description="Maximum number of evaluations to return")
):
    """List all evaluations."""
    evaluations = []
    
    for eval_id, eval_state in running_evaluations.items():
        if status and eval_state["status"] != status:
            continue
        
        evaluations.append({
            "evaluation_id": eval_id,
            "status": eval_state["status"],
            "started_at": eval_state["started_at"],
            "completed_at": eval_state.get("completed_at"),
            "models": eval_state["request"]["models"],
            "categories": eval_state["request"].get("categories")
        })
    
    # Sort by started_at descending
    evaluations.sort(key=lambda x: x["started_at"], reverse=True)
    
    return {"evaluations": evaluations[:limit]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
