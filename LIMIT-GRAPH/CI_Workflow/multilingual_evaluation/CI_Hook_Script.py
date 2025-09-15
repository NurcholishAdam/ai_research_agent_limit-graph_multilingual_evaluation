# -*- coding: utf-8 -*-
"""
CI Hook Script for Multilingual Agent Evaluation
Automates validation of agent submissions across languages and benchmark tasks
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from load_tests import MultilingualTestLoader, Language, TaskType
from run_agent import MultilingualAgentRunner, EvaluationMetrics

class CIHookRunner:
    """
    CI Hook for automated multilingual agent evaluation
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize CI Hook Runner
        
        Args:
            config_path: Path to CI configuration file
        """
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.test_loader = MultilingualTestLoader()
        self.agent_runner = MultilingualAgentRunner(self.test_loader)
        
        # CI-specific settings
        self.ci_environment = os.getenv('CI', 'false').lower() == 'true'
        self.github_actions = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'
        
        # Quality thresholds
        self.quality_thresholds = self.config.get('quality_thresholds', {
            'min_overall_score': 0.6,
            'min_accuracy': 0.7,
            'min_semantic_score': 0.6,
            'max_error_rate': 0.2,
            'min_rtl_score': 0.5,
            'min_cultural_awareness': 0.5
        })
        
        # Language requirements
        self.required_languages = [
            Language(lang) for lang in self.config.get('required_languages', ['en', 'id', 'ar'])
        ]
        
        # Results storage
        self.evaluation_results = {}
        self.leaderboard_path = self.config.get('leaderboard_path', 'leaderboard.json')
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for CI environment"""
        log_level = logging.DEBUG if os.getenv('CI_DEBUG') else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ci_evaluation.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return logging.getLogger(__name__)
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load CI configuration"""
        if config_path is None:
            config_path = Path(__file__).parent / 'ci_config.json'
        
        default_config = {
            'required_languages': ['en', 'id', 'ar'],
            'max_evaluation_time': 1800,  # 30 minutes
            'quality_thresholds': {
                'min_overall_score': 0.6,
                'min_accuracy': 0.7,
                'min_semantic_score': 0.6,
                'max_error_rate': 0.2
            },
            'leaderboard_path': 'leaderboard.json',
            'results_dir': 'evaluation_results',
            'enable_leaderboard_update': True,
            'enable_pr_comments': True
        }
        
        if Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Error loading config from {config_path}: {e}")
        
        return default_config
    
    def validate_agent_submission(self, agent_path: str, agent_name: str = None) -> Dict[str, Any]:
        """
        Validate agent submission against multilingual benchmarks
        
        Args:
            agent_path: Path to agent Python file
            agent_name: Name of the agent (for reporting)
            
        Returns:
            Validation results dictionary
        """
        if agent_name is None:
            agent_name = Path(agent_path).stem
        
        self.logger.info(f"Starting validation for agent: {agent_name}")
        self.logger.info(f"Agent path: {agent_path}")
        
        validation_result = {
            'agent_name': agent_name,
            'agent_path': agent_path,
            'validation_timestamp': time.time(),
            'validation_passed': False,
            'metrics': None,
            'quality_checks': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Step 1: Load and validate agent
            self.logger.info("Step 1: Loading agent")
            agent = self._load_and_validate_agent(agent_path)
            
            # Step 2: Run multilingual evaluation
            self.logger.info("Step 2: Running multilingual evaluation")
            metrics = self._run_comprehensive_evaluation(agent, agent_name)
            validation_result['metrics'] = asdict(metrics)
            
            # Step 3: Quality checks
            self.logger.info("Step 3: Performing quality checks")
            quality_checks = self._perform_quality_checks(metrics)
            validation_result['quality_checks'] = quality_checks
            
            # Step 4: Determine validation status
            validation_passed = all(quality_checks.values())
            validation_result['validation_passed'] = validation_passed
            
            # Step 5: Update leaderboard if validation passed
            if validation_passed and self.config.get('enable_leaderboard_update', True):
                self.logger.info("Step 5: Updating leaderboard")
                self._update_leaderboard(agent_name, metrics)
            
            # Step 6: Generate reports
            self.logger.info("Step 6: Generating reports")
            self._generate_evaluation_reports(agent_name, metrics, validation_result)
            
            if validation_passed:
                self.logger.info(f"✅ Validation PASSED for {agent_name}")
            else:
                self.logger.warning(f"❌ Validation FAILED for {agent_name}")
                
        except Exception as e:
            error_msg = f"Validation failed with error: {str(e)}"
            self.logger.error(error_msg)
            validation_result['errors'].append(error_msg)
            validation_result['validation_passed'] = False
        
        return validation_result
    
    def _load_and_validate_agent(self, agent_path: str) -> Any:
        """Load and validate agent structure"""
        if not Path(agent_path).exists():
            raise FileNotFoundError(f"Agent file not found: {agent_path}")
        
        # Load agent
        agent = self.agent_runner.load_agent(agent_path)
        
        # Validate agent interface
        if not (hasattr(agent, 'query') or hasattr(agent, 'generate') or hasattr(agent, '__call__')):
            raise ValueError("Agent must have 'query', 'generate', or '__call__' method")
        
        # Test basic functionality
        try:
            if hasattr(agent, 'query'):
                test_response = agent.query("Test question", "Test context")
            elif hasattr(agent, 'generate'):
                test_response = agent.generate("Test question", context="Test context")
            else:
                test_response = agent("Test question", context="Test context")
            
            if not isinstance(test_response, str):
                self.logger.warning("Agent response is not a string, will be converted")
                
        except Exception as e:
            raise ValueError(f"Agent failed basic functionality test: {e}")
        
        return agent
    
    def _run_comprehensive_evaluation(self, agent: Any, agent_name: str) -> EvaluationMetrics:
        """Run comprehensive multilingual evaluation"""
        
        # Set evaluation timeout
        max_time = self.config.get('max_evaluation_time', 1800)
        start_time = time.time()
        
        try:
            # Run evaluation with timeout consideration
            metrics = self.agent_runner.run_evaluation(
                agent=agent,
                languages=self.required_languages,
                max_tests_per_language=self.config.get('max_tests_per_language', 10)
            )
            
            evaluation_time = time.time() - start_time
            
            if evaluation_time > max_time:
                self.logger.warning(f"Evaluation took {evaluation_time:.1f}s (limit: {max_time}s)")
            
            return metrics
            
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {e}")
    
    def _perform_quality_checks(self, metrics: EvaluationMetrics) -> Dict[str, bool]:
        """Perform quality checks against thresholds"""
        checks = {}
        thresholds = self.quality_thresholds
        
        # Core quality checks
        checks['overall_score_check'] = metrics.overall_score >= thresholds.get('min_overall_score', 0.6)
        checks['accuracy_check'] = metrics.accuracy >= thresholds.get('min_accuracy', 0.7)
        checks['semantic_score_check'] = metrics.semantic_score >= thresholds.get('min_semantic_score', 0.6)
        checks['error_rate_check'] = metrics.error_rate <= thresholds.get('max_error_rate', 0.2)
        
        # Language-specific checks
        checks['rtl_handling_check'] = metrics.rtl_handling_score >= thresholds.get('min_rtl_score', 0.5)
        checks['cultural_awareness_check'] = metrics.cultural_awareness_score >= thresholds.get('min_cultural_awareness', 0.5)
        
        # Language coverage checks
        required_langs = [lang.value for lang in self.required_languages]
        available_langs = list(metrics.language_scores.keys())
        
        for lang in required_langs:
            lang_check_key = f'{lang}_language_check'
            if lang in available_langs:
                lang_score = metrics.language_scores[lang]
                checks[lang_check_key] = lang_score >= thresholds.get('min_language_score', 0.5)
            else:
                checks[lang_check_key] = False
        
        # Performance checks
        checks['response_time_check'] = metrics.response_time <= thresholds.get('max_response_time', 10.0)
        
        return checks
    
    def _update_leaderboard(self, agent_name: str, metrics: EvaluationMetrics):
        """Update leaderboard with new results"""
        leaderboard_path = Path(self.leaderboard_path)
        
        # Load existing leaderboard
        leaderboard = []
        if leaderboard_path.exists():
            try:
                with open(leaderboard_path, 'r') as f:
                    leaderboard = json.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading leaderboard: {e}")
                leaderboard = []
        
        # Create new entry
        new_entry = {
            'agent_name': agent_name,
            'timestamp': time.time(),
            'overall_score': metrics.overall_score,
            'accuracy': metrics.accuracy,
            'semantic_score': metrics.semantic_score,
            'distractor_robustness': metrics.distractor_robustness,
            'context_length_score': metrics.context_length_score,
            'rtl_handling_score': metrics.rtl_handling_score,
            'cultural_awareness_score': metrics.cultural_awareness_score,
            'cross_lingual_consistency': metrics.cross_lingual_consistency,
            'response_time': metrics.response_time,
            'error_rate': metrics.error_rate,
            'language_scores': metrics.language_scores,
            'task_scores': metrics.task_scores
        }
        
        # Remove existing entry for same agent (if any)
        leaderboard = [entry for entry in leaderboard if entry['agent_name'] != agent_name]
        
        # Add new entry
        leaderboard.append(new_entry)
        
        # Sort by overall score (descending)
        leaderboard.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Save updated leaderboard
        try:
            with open(leaderboard_path, 'w') as f:
                json.dump(leaderboard, f, indent=2)
            
            self.logger.info(f"Updated leaderboard: {agent_name} ranked #{leaderboard.index(new_entry) + 1}")
            
        except Exception as e:
            self.logger.error(f"Error updating leaderboard: {e}")
    
    def _generate_evaluation_reports(self, agent_name: str, metrics: EvaluationMetrics, 
                                   validation_result: Dict[str, Any]):
        """Generate evaluation reports"""
        results_dir = Path(self.config.get('results_dir', 'evaluation_results'))
        results_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Generate detailed JSON report
        json_report_path = results_dir / f"{agent_name}_{timestamp}_detailed.json"
        self.agent_runner.export_results(str(json_report_path), agent_name)
        
        # Generate human-readable report
        markdown_report = self.agent_runner.generate_evaluation_report(agent_name)
        
        # Add validation status to report
        validation_status = "✅ PASSED" if validation_result['validation_passed'] else "❌ FAILED"
        markdown_report = f"**Validation Status:** {validation_status}\n\n" + markdown_report
        
        # Add quality checks to report
        markdown_report += "\n## Quality Checks\n"
        for check_name, passed in validation_result['quality_checks'].items():
            status = "✅" if passed else "❌"
            check_display = check_name.replace('_', ' ').title()
            markdown_report += f"- {status} {check_display}\n"
        
        # Save markdown report
        md_report_path = results_dir / f"{agent_name}_{timestamp}_report.md"
        with open(md_report_path, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        
        # Generate CI summary for GitHub Actions
        if self.github_actions:
            self._generate_github_summary(agent_name, metrics, validation_result)
        
        self.logger.info(f"Generated reports: {json_report_path}, {md_report_path}")
    
    def _generate_github_summary(self, agent_name: str, metrics: EvaluationMetrics, 
                                validation_result: Dict[str, Any]):
        """Generate GitHub Actions job summary"""
        summary_file = os.getenv('GITHUB_STEP_SUMMARY')
        if not summary_file:
            return
        
        validation_status = "✅ PASSED" if validation_result['validation_passed'] else "❌ FAILED"
        
        summary = f"""
# Multilingual Agent Evaluation Results

**Agent:** {agent_name}
**Status:** {validation_status}

## Performance Metrics

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| Overall Score | {metrics.overall_score:.3f} | {self.quality_thresholds.get('min_overall_score', 0.6)} | {'✅' if metrics.overall_score >= self.quality_thresholds.get('min_overall_score', 0.6) else '❌'} |
| Accuracy | {metrics.accuracy:.3f} | {self.quality_thresholds.get('min_accuracy', 0.7)} | {'✅' if metrics.accuracy >= self.quality_thresholds.get('min_accuracy', 0.7) else '❌'} |
| Semantic Score | {metrics.semantic_score:.3f} | {self.quality_thresholds.get('min_semantic_score', 0.6)} | {'✅' if metrics.semantic_score >= self.quality_thresholds.get('min_semantic_score', 0.6) else '❌'} |
| Error Rate | {metrics.error_rate:.1%} | {self.quality_thresholds.get('max_error_rate', 0.2):.1%} | {'✅' if metrics.error_rate <= self.quality_thresholds.get('max_error_rate', 0.2) else '❌'} |

## Language Performance

| Language | Score |
|----------|-------|
"""
        
        for lang, score in metrics.language_scores.items():
            summary += f"| {lang.upper()} | {score:.3f} |\n"
        
        summary += f"""
## Specialized Metrics

- **Distractor Robustness:** {metrics.distractor_robustness:.3f}
- **Context Length Handling:** {metrics.context_length_score:.3f}
- **RTL Text Handling:** {metrics.rtl_handling_score:.3f}
- **Cultural Awareness:** {metrics.cultural_awareness_score:.3f}
- **Cross-lingual Consistency:** {metrics.cross_lingual_consistency:.3f}

## Quality Checks
"""
        
        for check_name, passed in validation_result['quality_checks'].items():
            status = "✅" if passed else "❌"
            check_display = check_name.replace('_', ' ').title()
            summary += f"- {status} {check_display}\n"
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
        except Exception as e:
            self.logger.error(f"Error writing GitHub summary: {e}")
    
    def run_ci_pipeline(self, agent_path: str, agent_name: str = None) -> bool:
        """
        Run complete CI pipeline for agent validation
        
        Args:
            agent_path: Path to agent file
            agent_name: Agent name (optional)
            
        Returns:
            True if validation passed, False otherwise
        """
        self.logger.info("🚀 Starting CI Pipeline for Multilingual Agent Evaluation")
        self.logger.info("="*70)
        
        try:
            # Run validation
            validation_result = self.validate_agent_submission(agent_path, agent_name)
            
            # Print summary
            if validation_result['validation_passed']:
                self.logger.info("🎉 CI Pipeline completed successfully!")
                self.logger.info(f"✅ Agent '{validation_result['agent_name']}' passed all quality checks")
                
                if validation_result['metrics']:
                    metrics = validation_result['metrics']
                    self.logger.info(f"📊 Overall Score: {metrics['overall_score']:.3f}")
                    self.logger.info(f"📊 Accuracy: {metrics['accuracy']:.3f}")
                    self.logger.info(f"📊 Semantic Score: {metrics['semantic_score']:.3f}")
                
            else:
                self.logger.error("❌ CI Pipeline failed!")
                self.logger.error(f"❌ Agent '{validation_result['agent_name']}' failed quality checks")
                
                failed_checks = [check for check, passed in validation_result['quality_checks'].items() if not passed]
                self.logger.error(f"Failed checks: {', '.join(failed_checks)}")
            
            return validation_result['validation_passed']
            
        except Exception as e:
            self.logger.error(f"💥 CI Pipeline crashed: {e}")
            return False

def main():
    """Main CLI interface for CI Hook"""
    parser = argparse.ArgumentParser(description="Multilingual Agent Evaluation CI Hook")
    
    parser.add_argument("agent_path", help="Path to agent Python file")
    parser.add_argument("--agent-name", help="Name of the agent")
    parser.add_argument("--config", help="Path to CI configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug environment variable
    if args.debug:
        os.environ['CI_DEBUG'] = 'true'
    
    # Initialize CI Hook
    ci_hook = CIHookRunner(args.config)
    
    # Run CI pipeline
    success = ci_hook.run_ci_pipeline(args.agent_path, args.agent_name)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()