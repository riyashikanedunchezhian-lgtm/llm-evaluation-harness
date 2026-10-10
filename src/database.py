"""SQLite persistence layer for evaluation results."""

import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from contextlib import contextmanager

from .judge import JudgeVerdict


@dataclass
class EvaluationRecord:
    """Database record for an evaluation result."""
    id: Optional[int]
    test_id: str
    category: str
    prompt: str
    model_id: str
    model_name: str
    response: str
    reference_answer: Optional[str]
    evaluation_criteria: Optional[str]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    judge_results_json: str  # JSON serialized judge results
    overall_score: float
    timestamp: str
    judge_input_tokens: int
    judge_output_tokens: int
    execution_mode: str
    created_at: str


class Database:
    """SQLite database for storing evaluation results."""

    def __init__(self, db_path: str = "data/evaluations.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self):
        """Initialize database schema."""
        with self._connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    response TEXT NOT NULL,
                    reference_answer TEXT,
                    evaluation_criteria TEXT,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    cost_usd REAL NOT NULL,
                    judge_results_json TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    judge_input_tokens INTEGER NOT NULL,
                    judge_output_tokens INTEGER NOT NULL,
                    execution_mode TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_evaluations_test_id ON evaluations(test_id);
                CREATE INDEX IF NOT EXISTS idx_evaluations_model_id ON evaluations(model_id);
                CREATE INDEX IF NOT EXISTS idx_evaluations_category ON evaluations(category);
                CREATE INDEX IF NOT EXISTS idx_evaluations_timestamp ON evaluations(timestamp);

                CREATE TABLE IF NOT EXISTS bias_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response_a TEXT NOT NULL,
                    response_b TEXT NOT NULL,
                    reference_answer TEXT,
                    evaluation_criteria TEXT,
                    bias_detected INTEGER NOT NULL,
                    score_delta_a REAL NOT NULL,
                    score_delta_b REAL NOT NULL,
                    score_a_ab REAL NOT NULL,
                    score_a_ba REAL NOT NULL,
                    score_b_ab REAL NOT NULL,
                    score_b_ba REAL NOT NULL,
                    details_json TEXT,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_bias_checks_test_id ON bias_checks(test_id);
                CREATE INDEX IF NOT EXISTS idx_bias_checks_timestamp ON bias_checks(timestamp);

                CREATE TABLE IF NOT EXISTS aggregated_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_type TEXT NOT NULL,
                    metric_key TEXT NOT NULL,
                    metric_value_json TEXT NOT NULL,
                    test_run_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_aggregated_metrics_type ON aggregated_metrics(metric_type);
                CREATE INDEX IF NOT EXISTS idx_aggregated_metrics_key ON aggregated_metrics(metric_key);
            """)
            conn.commit()

    def save_evaluation(self, result) -> int:
        """Save a single evaluation result. Returns the row ID."""
        # Handle EvaluationResult dataclass or dict
        if hasattr(result, '__dataclass_fields__'):
            data = asdict(result)
        else:
            data = result

        # Serialize judge_results
        judge_results_json = json.dumps(data.get('judge_results', {}), default=str)

        with self._connection() as conn:
            cursor = conn.execute("""
                INSERT INTO evaluations (
                    test_id, category, prompt, model_id, model_name, response,
                    reference_answer, evaluation_criteria, input_tokens, output_tokens,
                    latency_ms, cost_usd, judge_results_json, overall_score,
                    timestamp, judge_input_tokens, judge_output_tokens, execution_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['test_id'],
                data['category'],
                data['prompt'],
                data['model_id'],
                data['model_name'],
                data['response'],
                data.get('reference_answer'),
                data.get('evaluation_criteria'),
                data['input_tokens'],
                data['output_tokens'],
                data['latency_ms'],
                data['cost_usd'],
                judge_results_json,
                data['overall_score'],
                data['timestamp'],
                data.get('judge_input_tokens', 0),
                data.get('judge_output_tokens', 0),
                data.get('execution_mode', 'sequential')
            ))
            conn.commit()
            return cursor.lastrowid

    def save_evaluations_batch(self, results: List) -> List[int]:
        """Save multiple evaluation results efficiently."""
        ids = []
        with self._connection() as conn:
            for result in results:
                if hasattr(result, '__dataclass_fields__'):
                    data = asdict(result)
                else:
                    data = result

                judge_results_json = json.dumps(data.get('judge_results', {}), default=str)

                cursor = conn.execute("""
                    INSERT INTO evaluations (
                        test_id, category, prompt, model_id, model_name, response,
                        reference_answer, evaluation_criteria, input_tokens, output_tokens,
                        latency_ms, cost_usd, judge_results_json, overall_score,
                        timestamp, judge_input_tokens, judge_output_tokens, execution_mode
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data['test_id'],
                    data['category'],
                    data['prompt'],
                    data['model_id'],
                    data['model_name'],
                    data['response'],
                    data.get('reference_answer'),
                    data.get('evaluation_criteria'),
                    data['input_tokens'],
                    data['output_tokens'],
                    data['latency_ms'],
                    data['cost_usd'],
                    judge_results_json,
                    data['overall_score'],
                    data['timestamp'],
                    data.get('judge_input_tokens', 0),
                    data.get('judge_output_tokens', 0),
                    data.get('execution_mode', 'sequential')
                ))
                ids.append(cursor.lastrowid)
            conn.commit()
        return ids

    def get_evaluation(self, evaluation_id: int) -> Optional[Dict]:
        """Get a single evaluation by ID."""
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)).fetchone()
            if row:
                return dict(row)
            return None

    def get_evaluations(
        self,
        test_id: Optional[str] = None,
        model_id: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """Get evaluations with optional filters."""
        query = "SELECT * FROM evaluations WHERE 1=1"
        params = []

        if test_id:
            query += " AND test_id = ?"
            params.append(test_id)
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        if category:
            query += " AND category = ?"
            params.append(category)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def save_bias_check(self, bias_result: Dict) -> int:
        """Save a position bias check result."""
        with self._connection() as conn:
            cursor = conn.execute("""
                INSERT INTO bias_checks (
                    test_id, prompt, response_a, response_b, reference_answer,
                    evaluation_criteria, bias_detected, score_delta_a, score_delta_b,
                    score_a_ab, score_a_ba, score_b_ab, score_b_ba, details_json, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                bias_result['test_id'],
                bias_result['prompt'],
                bias_result['response_a'],
                bias_result['response_b'],
                bias_result.get('reference_answer'),
                bias_result.get('evaluation_criteria'),
                1 if bias_result['bias_detected'] else 0,
                bias_result['score_delta_a'],
                bias_result['score_delta_b'],
                bias_result['score_a_ab'],
                bias_result['score_a_ba'],
                bias_result['score_b_ab'],
                bias_result['score_b_ba'],
                json.dumps(bias_result.get('details', {}), default=str),
                bias_result['timestamp']
            ))
            conn.commit()
            return cursor.lastrowid

    def get_bias_checks(
        self,
        test_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get bias checks with optional filter."""
        query = "SELECT * FROM bias_checks WHERE 1=1"
        params = []

        if test_id:
            query += " AND test_id = ?"
            params.append(test_id)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def save_aggregated_metrics(
        self,
        metric_type: str,
        metric_key: str,
        metric_value: Dict,
        test_run_id: Optional[str] = None
    ) -> int:
        """Save aggregated metrics."""
        with self._connection() as conn:
            cursor = conn.execute("""
                INSERT INTO aggregated_metrics (metric_type, metric_key, metric_value_json, test_run_id)
                VALUES (?, ?, ?, ?)
            """, (metric_type, metric_key, json.dumps(metric_value, default=str), test_run_id))
            conn.commit()
            return cursor.lastrowid

    def get_aggregated_metrics(
        self,
        metric_type: Optional[str] = None,
        metric_key: Optional[str] = None
    ) -> List[Dict]:
        """Get aggregated metrics with optional filters."""
        query = "SELECT * FROM aggregated_metrics WHERE 1=1"
        params = []

        if metric_type:
            query += " AND metric_type = ?"
            params.append(metric_type)
        if metric_key:
            query += " AND metric_key = ?"
            params.append(metric_key)

        query += " ORDER BY created_at DESC"

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from the database."""
        with self._connection() as conn:
            # Total evaluations
            total = conn.execute("SELECT COUNT(*) as count FROM evaluations").fetchone()['count']

            # By model
            by_model = conn.execute("""
                SELECT model_id, model_name, COUNT(*) as count,
                       AVG(overall_score) as avg_score,
                       AVG(latency_ms) as avg_latency,
                       SUM(cost_usd) as total_cost
                FROM evaluations
                GROUP BY model_id, model_name
            """).fetchall()

            # By category
            by_category = conn.execute("""
                SELECT category, COUNT(*) as count,
                       AVG(overall_score) as avg_score,
                       AVG(latency_ms) as avg_latency
                FROM evaluations
                GROUP BY category
            """).fetchall()

            # Total cost
            total_cost = conn.execute("SELECT SUM(cost_usd) as total FROM evaluations").fetchone()['total'] or 0

            # Total tokens
            total_tokens = conn.execute("""
                SELECT SUM(input_tokens + output_tokens + judge_input_tokens + judge_output_tokens) as total
                FROM evaluations
            """).fetchone()['total'] or 0

            return {
                "total_evaluations": total,
                "total_cost_usd": total_cost,
                "total_tokens": total_tokens,
                "by_model": [dict(row) for row in by_model],
                "by_category": [dict(row) for row in by_category]
            }

    def export_to_json(self, output_path: str, filters: Optional[Dict] = None):
        """Export evaluations to JSON file."""
        evaluations = self.get_evaluations(
            test_id=filters.get('test_id') if filters else None,
            model_id=filters.get('model_id') if filters else None,
            category=filters.get('category') if filters else None,
            limit=10000
        )

        # Parse judge_results_json back to dict
        for eval in evaluations:
            eval['judge_results'] = json.loads(eval['judge_results_json'])

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluations, f, indent=2, default=str)

        return len(evaluations)


# Convenience function for quick access
_default_db: Optional[Database] = None

def get_database(db_path: str = "data/evaluations.db") -> Database:
    """Get the default database instance (singleton pattern)."""
    global _default_db
    if _default_db is None:
        _default_db = Database(db_path)
    return _default_db