# -*- coding: utf-8 -*-
"""
Agent Evaluator Runner for Multilingual Agent Evaluation
Runs agents against multilingual benchmark tests and computes comprehensive metrics
"""

import json
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import importlib.util
import sys

from load_tests import MultilingualTestLoader, TestCase, Language, TaskType

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for multilingual agents"""
    # Core Metrics
    accuracy: float
    semantic_score: float
    distractor_robustness: float
    context_length_score: float
    
    # Language-specific Metrics
    rtl_handling_score: float
    cultural_awareness_score: float
    cross_lingual_consistency: float
    
    # Performance Metrics
    response_time: float
    token_efficiency: float
    error_rate: float
    
    # Detailed Scores
    task_scores: Dict[str, float]
    language_scores: Dict[str, float]
    difficulty_scores: Dict[str, float]
    
    # Overall Score
    overall_score: float

@dataclass
class TestResult:
    """Individual test result"""
    test_id: str
    language: str
    task_type: str
    input_text: str
    expected_output: str
    agent_output: str
    context: str
    
    # Scores
    accuracy_score: float
    semantic_score: float
    response_time: float
    
    # Metadata
    difficulty: str
    cultural_context: Optional[str]
    rtl_specific: bool
    error_message: Optional[str] = None

class MultilingualAgentRunner:
    """
    Runs multilingual agents against benchmark tests and computes metrics
    """
    
    def __init__(self, test_loader: MultilingualTestLoader = None):
        """
        Initialize agent runner
        
        Args:
            test_loader: Test loader instance
        """
        self.logger = logging.getLogger(__name__)
        self.test_loader = test_loader or MultilingualTestLoader()
        
        # Initialize TF-IDF vectorizer for semantic similarity
        self.vectorizer = TfidfVectorizer(
            stop_words=None,  # Keep all words for multilingual support
            lowercase=True,
            ngram_range=(1, 2),
            max_features=10000
        )
        
        # Evaluation results storage
        self.evaluation_results = []
        
        # Metric weights for overall score calculation
        self.metric_weights = {
            "accuracy": 0.25,
            "semantic_score": 0.20,
            "distractor_robustness": 0.15,
            "context_length_score": 0.15,
            "rtl_handling_score": 0.10,
            "cultural_awareness_score": 0.10,
            "cross_lingual_consistency": 0.05
        }
    
    def load_agent(self, agent_path: str) -> Any:
        """
        Load agent from Python file
        
        Args:
            agent_path: Path to agent Python file
            
        Returns:
            Loaded agent instance
        """
        try:
            # Load agent module
            spec = importlib.util.spec_from_file_location("agent_module", agent_path)
            agent_module = importlib.util.module_from_spec(spec)
            sys.modules["agent_module"] = agent_module
            spec.loader.exec_module(agent_module)
            
            # Look for agent class or function
            if hasattr(agent_module, 'Agent'):
                agent = agent_module.Agent()
            elif hasattr(agent_module, 'MultilingualAgent'):
                agent = agent_module.MultilingualAgent()
            elif hasattr(agent_module, 'agent'):
                agent = agent_module.agent
            else:
                # Try to find any class that has a 'query' or 'generate' method
                for attr_name in dir(agent_module):
                    attr = getattr(agent_module, attr_name)
                    if (hasattr(attr, '__call__') and 
                        (hasattr(attr, 'query') or hasattr(attr, 'generate'))):
                        agent = attr()
                        break
                else:
                    raise ValueError("No suitable agent class found in module")
            
            self.logger.info(f"Successfully loaded agent from {agent_path}")
            return agent
            
        except Exception as e:
            self.logger.error(f"Error loading agent from {agent_path}: {e}")
            raise
    
    def run_single_test(self, agent: Any, test_case: TestCase) -> TestResult:
        """
        Run agent on a single test case
        
        Args:
            agent: Agent instance
            test_case: Test case to run
            
        Returns:
            TestResult with scores and outputs
        """
        start_time = time.time()
        error_message = None
        agent_output = ""
        
        try:
            # Prepare input for agent
            if hasattr(agent, 'query'):
                # Standard query interface
                if test_case.context:
                    agent_output = agent.query(test_case.input_text, test_case.context)
                else:
                    agent_output = agent.query(test_case.input_text)
            elif hasattr(agent, 'generate'):
                # Generation interface
                agent_output = agent.generate(test_case.input_text, context=test_case.context)
            elif hasattr(agent, '__call__'):
                # Callable interface
                agent_output = agent(test_case.input_text, context=test_case.context)
            else:
                raise ValueError("Agent does not have a supported interface (query, generate, or __call__)")
            
            # Ensure output is string
            if not isinstance(agent_output, str):
                agent_output = str(agent_output)
                
        except Exception as e:
            error_message = str(e)
            agent_output = f"ERROR: {error_message}"
            self.logger.error(f"Error running test {test_case.id}: {e}")
        
        response_time = time.time() - start_time
        
        # Calculate scores
        accuracy_score = self._calculate_accuracy(agent_output, test_case.expected_output, error_message)
        semantic_score = self._calculate_semantic_similarity(agent_output, test_case.expected_output)
        
        return TestResult(
            test_id=test_case.id,
            language=test_case.language,
            task_type=test_case.task_type,
            input_text=test_case.input_text,
            expected_output=test_case.expected_output,
            agent_output=agent_output,
            context=test_case.context,
            accuracy_score=accuracy_score,
            semantic_score=semantic_score,
            response_time=response_time,
            difficulty=test_case.difficulty,
            cultural_context=test_case.cultural_context,
            rtl_specific=test_case.rtl_specific,
            error_message=error_message
        )
    
    def _calculate_accuracy(self, agent_output: str, expected_output: str, error_message: Optional[str]) -> float:
        """Calculate accuracy score"""
        if error_message:
            return 0.0
        
        # Normalize outputs for comparison
        agent_normalized = agent_output.lower().strip()
        expected_normalized = expected_output.lower().strip()
        
        # Exact match
        if agent_normalized == expected_normalized:
            return 1.0
        
        # Partial match - check if expected is contained in agent output
        if expected_normalized in agent_normalized:
            return 0.8
        
        # Check if agent output contains key terms from expected
        expected_words = set(expected_normalized.split())
        agent_words = set(agent_normalized.split())
        
        if expected_words and agent_words:
            overlap = len(expected_words.intersection(agent_words))
            return min(0.7, overlap / len(expected_words))
        
        return 0.0
    
    def _calculate_semantic_similarity(self, agent_output: str, expected_output: str) -> float:
        """Calculate semantic similarity using TF-IDF and cosine similarity"""
        try:
            if not agent_output.strip() or not expected_output.strip():
                return 0.0
            
            # Vectorize texts
            texts = [agent_output, expected_output]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            similarity_score = similarity_matrix[0, 1]
            
            return max(0.0, min(1.0, similarity_score))
            
        except Exception as e:
            self.logger.warning(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def run_evaluation(self, agent: Any, languages: List[Language] = None, 
                      task_types: List[TaskType] = None,
                      max_tests_per_language: int = None) -> EvaluationMetrics:
        """
        Run comprehensive evaluation on agent
        
        Args:
            agent: Agent instance to evaluate
            languages: Languages to test (default: all)
            task_types: Task types to test (default: all)
            max_tests_per_language: Maximum tests per language
            
        Returns:
            EvaluationMetrics with comprehensive scores
        """
        self.logger.info("Starting comprehensive multilingual evaluation")
        
        if languages is None:
            languages = list(Language)
        
        if task_types is None:
            task_types = list(TaskType)
        
        # Load test suites
        all_results = []
        
        for language in languages:
            self.logger.info(f"Evaluating {language.value}")
            
            # Load language tests
            test_suite = self.test_loader.load_language_tests(language)
            
            # Filter by task types if specified
            language_tests = test_suite.test_cases
            if task_types:
                language_tests = [test for test in language_tests 
                                if TaskType(test.task_type) in task_types]
            
            # Limit number of tests if specified
            if max_tests_per_language:
                language_tests = language_tests[:max_tests_per_language]
            
            # Run tests
            for test_case in language_tests:
                result = self.run_single_test(agent, test_case)
                all_results.append(result)
                
                self.logger.debug(f"Test {result.test_id}: Accuracy={result.accuracy_score:.3f}, "
                                f"Semantic={result.semantic_score:.3f}")
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(all_results)
        
        # Store results
        self.evaluation_results = all_results
        
        self.logger.info(f"Evaluation completed: Overall Score = {metrics.overall_score:.3f}")
        
        return metrics
    
    def _calculate_comprehensive_metrics(self, results: List[TestResult]) -> EvaluationMetrics:
        """Calculate comprehensive evaluation metrics"""
        if not results:
            return EvaluationMetrics(
                accuracy=0.0, semantic_score=0.0, distractor_robustness=0.0,
                context_length_score=0.0, rtl_handling_score=0.0,
                cultural_awareness_score=0.0, cross_lingual_consistency=0.0,
                response_time=0.0, token_efficiency=0.0, error_rate=1.0,
                task_scores={}, language_scores={}, difficulty_scores={},
                overall_score=0.0
            )
        
        # Core metrics
        accuracy = np.mean([r.accuracy_score for r in results])
        semantic_score = np.mean([r.semantic_score for r in results])
        response_time = np.mean([r.response_time for r in results])
        error_rate = len([r for r in results if r.error_message]) / len(results)
        
        # Distractor robustness (based on performance on difficult tasks)
        distractor_results = [r for r in results if r.task_type == "distractor_resistance"]
        distractor_robustness = np.mean([r.accuracy_score for r in distractor_results]) if distractor_results else accuracy
        
        # Context length handling
        context_results = [r for r in results if r.task_type == "context_length"]
        context_length_score = np.mean([r.accuracy_score for r in context_results]) if context_results else accuracy
        
        # RTL handling (Arabic-specific)
        rtl_results = [r for r in results if r.rtl_specific]
        rtl_handling_score = np.mean([r.accuracy_score for r in rtl_results]) if rtl_results else accuracy
        
        # Cultural awareness
        cultural_results = [r for r in results if r.cultural_context]
        cultural_awareness_score = np.mean([r.accuracy_score for r in cultural_results]) if cultural_results else accuracy
        
        # Cross-lingual consistency
        cross_lingual_consistency = self._calculate_cross_lingual_consistency(results)
        
        # Token efficiency (simplified)
        token_efficiency = 1.0 - min(1.0, response_time / 5.0)  # Assume 5s is slow
        
        # Aggregate scores by categories
        task_scores = {}
        for task_type in set(r.task_type for r in results):
            task_results = [r for r in results if r.task_type == task_type]
            task_scores[task_type] = np.mean([r.accuracy_score for r in task_results])
        
        language_scores = {}
        for language in set(r.language for r in results):
            lang_results = [r for r in results if r.language == language]
            language_scores[language] = np.mean([r.accuracy_score for r in lang_results])
        
        difficulty_scores = {}
        for difficulty in set(r.difficulty for r in results):
            diff_results = [r for r in results if r.difficulty == difficulty]
            difficulty_scores[difficulty] = np.mean([r.accuracy_score for r in diff_results])
        
        # Calculate overall score
        overall_score = (
            self.metric_weights["accuracy"] * accuracy +
            self.metric_weights["semantic_score"] * semantic_score +
            self.metric_weights["distractor_robustness"] * distractor_robustness +
            self.metric_weights["context_length_score"] * context_length_score +
            self.metric_weights["rtl_handling_score"] * rtl_handling_score +
            self.metric_weights["cultural_awareness_score"] * cultural_awareness_score +
            self.metric_weights["cross_lingual_consistency"] * cross_lingual_consistency
        )
        
        return EvaluationMetrics(
            accuracy=accuracy,
            semantic_score=semantic_score,
            distractor_robustness=distractor_robustness,
            context_length_score=context_length_score,
            rtl_handling_score=rtl_handling_score,
            cultural_awareness_score=cultural_awareness_score,
            cross_lingual_consistency=cross_lingual_consistency,
            response_time=response_time,
            token_efficiency=token_efficiency,
            error_rate=error_rate,
            task_scores=task_scores,
            language_scores=language_scores,
            difficulty_scores=difficulty_scores,
            overall_score=overall_score
        )
    
    def _calculate_cross_lingual_consistency(self, results: List[TestResult]) -> float:
        """Calculate cross-lingual consistency score"""
        # Group results by similar test content across languages
        language_groups = {}
        for result in results:
            # Simple grouping by task type and difficulty
            group_key = f"{result.task_type}_{result.difficulty}"
            if group_key not in language_groups:
                language_groups[group_key] = {}
            
            language_groups[group_key][result.language] = result.accuracy_score
        
        # Calculate consistency within each group
        consistency_scores = []
        for group_key, lang_scores in language_groups.items():
            if len(lang_scores) > 1:
                scores = list(lang_scores.values())
                # Consistency is inverse of standard deviation
                consistency = 1.0 - min(1.0, np.std(scores))
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def run_distractor_robustness_test(self, agent: Any, base_tests: List[TestCase]) -> float:
        """
        Run specific distractor robustness test
        
        Args:
            agent: Agent to test
            base_tests: Base test cases to add distractors to
            
        Returns:
            Distractor robustness score
        """
        # This would integrate with the distractor injection framework
        # For now, return a placeholder based on difficult tests
        difficult_tests = [test for test in base_tests if test.difficulty == "hard"]
        
        if not difficult_tests:
            return 0.5  # Default score if no difficult tests
        
        robustness_scores = []
        for test in difficult_tests[:5]:  # Test on first 5 difficult tests
            result = self.run_single_test(agent, test)
            robustness_scores.append(result.accuracy_score)
        
        return np.mean(robustness_scores)
    
    def run_context_length_test(self, agent: Any, max_length: int = 4000) -> Tuple[float, int]:
        """
        Test agent's context length handling capability
        
        Args:
            agent: Agent to test
            max_length: Maximum context length to test
            
        Returns:
            Tuple of (score, max_tokens_handled)
        """
        # Generate test with increasing context lengths
        base_context = "This is a test context. " * 50  # Base context
        test_question = "What is this text about?"
        expected_answer = "test context"
        
        context_lengths = [100, 500, 1000, 2000, max_length]
        scores = []
        max_handled = 0
        
        for length in context_lengths:
            # Create context of specified length
            context = (base_context * (length // len(base_context) + 1))[:length]
            
            try:
                start_time = time.time()
                
                if hasattr(agent, 'query'):
                    response = agent.query(test_question, context)
                else:
                    response = agent(test_question, context=context)
                
                response_time = time.time() - start_time
                
                # Check if response is reasonable (not empty, not error, reasonable time)
                if (response and 
                    len(response.strip()) > 0 and 
                    "error" not in response.lower() and 
                    response_time < 30.0):  # 30 second timeout
                    
                    max_handled = length
                    # Score based on response quality and time
                    quality_score = 1.0 if expected_answer.lower() in response.lower() else 0.5
                    time_penalty = max(0.0, 1.0 - response_time / 10.0)  # Penalty for slow responses
                    scores.append(quality_score * time_penalty)
                else:
                    scores.append(0.0)
                    break  # Stop testing longer contexts if this one failed
                    
            except Exception as e:
                self.logger.warning(f"Context length test failed at {length} tokens: {e}")
                scores.append(0.0)
                break
        
        # Calculate overall context length score
        if scores:
            context_score = np.mean(scores)
        else:
            context_score = 0.0
        
        return context_score, max_handled
    
    def export_results(self, output_path: str, agent_name: str = "Unknown"):
        """Export evaluation results to JSON file"""
        if not self.evaluation_results:
            self.logger.warning("No evaluation results to export")
            return
        
        # Calculate metrics
        metrics = self._calculate_comprehensive_metrics(self.evaluation_results)
        
        export_data = {
            "agent_name": agent_name,
            "evaluation_timestamp": time.time(),
            "metrics": asdict(metrics),
            "detailed_results": []
        }
        
        # Add detailed results
        for result in self.evaluation_results:
            export_data["detailed_results"].append(asdict(result))
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Exported evaluation results to {output_path}")
    
    def generate_evaluation_report(self, agent_name: str = "Unknown") -> str:
        """Generate human-readable evaluation report"""
        if not self.evaluation_results:
            return "No evaluation results available"
        
        metrics = self._calculate_comprehensive_metrics(self.evaluation_results)
        
        report = f"""
# Multilingual Agent Evaluation Report

**Agent:** {agent_name}
**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Total Tests:** {len(self.evaluation_results)}

## Overall Performance
- **Overall Score:** {metrics.overall_score:.3f}
- **Accuracy:** {metrics.accuracy:.3f}
- **Semantic Score:** {metrics.semantic_score:.3f}
- **Response Time:** {metrics.response_time:.2f}s
- **Error Rate:** {metrics.error_rate:.1%}

## Specialized Metrics
- **Distractor Robustness:** {metrics.distractor_robustness:.3f}
- **Context Length Handling:** {metrics.context_length_score:.3f}
- **RTL Text Handling:** {metrics.rtl_handling_score:.3f}
- **Cultural Awareness:** {metrics.cultural_awareness_score:.3f}
- **Cross-lingual Consistency:** {metrics.cross_lingual_consistency:.3f}

## Performance by Language
"""
        
        for language, score in metrics.language_scores.items():
            report += f"- **{language.upper()}:** {score:.3f}\n"
        
        report += "\n## Performance by Task Type\n"
        for task, score in metrics.task_scores.items():
            report += f"- **{task.replace('_', ' ').title()}:** {score:.3f}\n"
        
        report += "\n## Performance by Difficulty\n"
        for difficulty, score in metrics.difficulty_scores.items():
            report += f"- **{difficulty.title()}:** {score:.3f}\n"
        
        return report

def main():
    """Demo function for agent runner"""
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple demo agent
    class DemoAgent:
        def query(self, question: str, context: str = "") -> str:
            # Simple demo agent that returns basic responses
            question_lower = question.lower()
            
            if "capital" in question_lower and "france" in question_lower:
                return "Paris"
            elif "capital" in question_lower and "indonesia" in question_lower:
                return "Jakarta"
            elif "عاصمة" in question and "السعودية" in question:
                return "الرياض"
            elif "capital" in question_lower and "spain" in question_lower:
                return "Madrid"
            else:
                return "I don't know the answer to that question."
    
    # Initialize runner and test
    runner = MultilingualAgentRunner()
    demo_agent = DemoAgent()
    
    # Run evaluation
    metrics = runner.run_evaluation(
        demo_agent, 
        languages=[Language.ENGLISH, Language.INDONESIAN, Language.ARABIC],
        max_tests_per_language=3
    )
    
    # Print results
    print("Demo Agent Evaluation Results:")
    print(f"Overall Score: {metrics.overall_score:.3f}")
    print(f"Accuracy: {metrics.accuracy:.3f}")
    print(f"Semantic Score: {metrics.semantic_score:.3f}")
    
    # Generate report
    report = runner.generate_evaluation_report("DemoAgent")
    print(report)

if __name__ == "__main__":
    main()