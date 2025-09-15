# -*- coding: utf-8 -*-
"""
Language-Specific Test Loader for Multilingual Agent Evaluation
Loads and manages benchmark tests across multiple languages
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class Language(Enum):
    """Supported languages for evaluation"""
    ENGLISH = "en"
    INDONESIAN = "id"
    SPANISH = "es"
    ARABIC = "ar"
    FRENCH = "fr"

class TaskType(Enum):
    """Types of evaluation tasks"""
    NEEDLE_HAYSTACK = "needle_haystack"
    QUESTION_ANSWERING = "question_answering"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    DISTRACTOR_RESISTANCE = "distractor_resistance"
    CONTEXT_LENGTH = "context_length"
    CULTURAL_AWARENESS = "cultural_awareness"
    RTL_PROCESSING = "rtl_processing"

@dataclass
class TestCase:
    """Individual test case structure"""
    id: str
    language: str
    task_type: str
    input_text: str
    expected_output: str
    context: str
    metadata: Dict[str, Any]
    difficulty: str
    cultural_context: Optional[str] = None
    rtl_specific: bool = False

@dataclass
class TestSuite:
    """Collection of test cases for a language"""
    language: str
    test_cases: List[TestCase]
    metadata: Dict[str, Any]
    total_tests: int
    difficulty_distribution: Dict[str, int]

class MultilingualTestLoader:
    """
    Loads and manages multilingual test suites for agent evaluation
    """
    
    def __init__(self, test_data_dir: str = None):
        """
        Initialize test loader
        
        Args:
            test_data_dir: Directory containing test data files
        """
        self.logger = logging.getLogger(__name__)
        
        if test_data_dir is None:
            test_data_dir = Path(__file__).parent / "tests"
        
        self.test_data_dir = Path(test_data_dir)
        self.test_suites = {}
        
        # Language-specific configurations
        self.language_configs = {
            Language.ENGLISH: {
                "name": "English",
                "script": "latin",
                "rtl": False,
                "cultural_contexts": ["western", "academic", "business", "technical"]
            },
            Language.INDONESIAN: {
                "name": "Indonesian",
                "script": "latin",
                "rtl": False,
                "cultural_contexts": ["southeast_asian", "islamic", "modern", "traditional"]
            },
            Language.SPANISH: {
                "name": "Spanish",
                "script": "latin",
                "rtl": False,
                "cultural_contexts": ["hispanic", "latin_american", "european", "modern"]
            },
            Language.ARABIC: {
                "name": "Arabic",
                "script": "arabic",
                "rtl": True,
                "cultural_contexts": ["islamic", "middle_eastern", "classical", "modern"]
            },
            Language.FRENCH: {
                "name": "French",
                "script": "latin",
                "rtl": False,
                "cultural_contexts": ["francophone", "european", "african", "modern"]
            }
        }
        
        # Initialize test data directory structure
        self._initialize_test_structure()
    
    def _initialize_test_structure(self):
        """Initialize test directory structure"""
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        for language in Language:
            lang_dir = self.test_data_dir / language.value
            lang_dir.mkdir(exist_ok=True)
            
            # Create task-specific subdirectories
            for task_type in TaskType:
                task_dir = lang_dir / task_type.value
                task_dir.mkdir(exist_ok=True)
    
    def load_language_tests(self, language: Language) -> TestSuite:
        """
        Load all tests for a specific language
        
        Args:
            language: Language to load tests for
            
        Returns:
            TestSuite containing all tests for the language
        """
        self.logger.info(f"Loading tests for {language.value}")
        
        lang_dir = self.test_data_dir / language.value
        test_cases = []
        
        if not lang_dir.exists():
            self.logger.warning(f"No test directory found for {language.value}")
            return TestSuite(
                language=language.value,
                test_cases=[],
                metadata={},
                total_tests=0,
                difficulty_distribution={}
            )
        
        # Load tests from each task type
        for task_type in TaskType:
            task_dir = lang_dir / task_type.value
            if task_dir.exists():
                task_tests = self._load_task_tests(language, task_type, task_dir)
                test_cases.extend(task_tests)
        
        # If no test files exist, generate default tests
        if not test_cases:
            test_cases = self._generate_default_tests(language)
        
        # Calculate metadata
        difficulty_distribution = {}
        for test_case in test_cases:
            difficulty = test_case.difficulty
            difficulty_distribution[difficulty] = difficulty_distribution.get(difficulty, 0) + 1
        
        test_suite = TestSuite(
            language=language.value,
            test_cases=test_cases,
            metadata=self.language_configs[language],
            total_tests=len(test_cases),
            difficulty_distribution=difficulty_distribution
        )
        
        self.test_suites[language.value] = test_suite
        self.logger.info(f"Loaded {len(test_cases)} tests for {language.value}")
        
        return test_suite
    
    def _load_task_tests(self, language: Language, task_type: TaskType, task_dir: Path) -> List[TestCase]:
        """Load tests for a specific task type"""
        test_cases = []
        
        # Look for JSON files in the task directory
        for test_file in task_dir.glob("*.json"):
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    test_data = json.load(f)
                
                # Handle both single test case and list of test cases
                if isinstance(test_data, list):
                    for item in test_data:
                        test_case = self._create_test_case(language, task_type, item)
                        if test_case:
                            test_cases.append(test_case)
                else:
                    test_case = self._create_test_case(language, task_type, test_data)
                    if test_case:
                        test_cases.append(test_case)
                        
            except Exception as e:
                self.logger.error(f"Error loading test file {test_file}: {e}")
        
        return test_cases
    
    def _create_test_case(self, language: Language, task_type: TaskType, data: Dict[str, Any]) -> Optional[TestCase]:
        """Create a TestCase object from data"""
        try:
            test_case = TestCase(
                id=data.get("id", f"{language.value}_{task_type.value}_{len(data)}"),
                language=language.value,
                task_type=task_type.value,
                input_text=data.get("input", data.get("question", "")),
                expected_output=data.get("expected", data.get("answer", "")),
                context=data.get("context", ""),
                metadata=data.get("metadata", {}),
                difficulty=data.get("difficulty", "medium"),
                cultural_context=data.get("cultural_context"),
                rtl_specific=data.get("rtl_specific", language == Language.ARABIC)
            )
            return test_case
        except Exception as e:
            self.logger.error(f"Error creating test case: {e}")
            return None
    
    def _generate_default_tests(self, language: Language) -> List[TestCase]:
        """Generate default test cases for a language"""
        self.logger.info(f"Generating default tests for {language.value}")
        
        default_tests = []
        
        # Language-specific test templates
        test_templates = {
            Language.ENGLISH: {
                "needle_haystack": [
                    {
                        "input": "What is the capital of France?",
                        "expected": "Paris",
                        "context": "France is a country in Europe. Paris is the capital and largest city of France. The city is known for the Eiffel Tower.",
                        "difficulty": "easy"
                    },
                    {
                        "input": "Who developed the theory of relativity?",
                        "expected": "Albert Einstein",
                        "context": "Albert Einstein was a theoretical physicist. He developed the theory of relativity. Einstein won the Nobel Prize in Physics.",
                        "difficulty": "medium"
                    }
                ],
                "question_answering": [
                    {
                        "input": "Explain the concept of machine learning.",
                        "expected": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                        "context": "Technology and AI concepts",
                        "difficulty": "medium"
                    }
                ]
            },
            Language.INDONESIAN: {
                "needle_haystack": [
                    {
                        "input": "Apa ibu kota Indonesia?",
                        "expected": "Jakarta",
                        "context": "Indonesia adalah negara kepulauan di Asia Tenggara. Jakarta adalah ibu kota dan kota terbesar Indonesia. Kota ini terletak di Pulau Jawa.",
                        "difficulty": "easy"
                    },
                    {
                        "input": "Siapa presiden pertama Indonesia?",
                        "expected": "Soekarno",
                        "context": "Soekarno adalah presiden pertama Republik Indonesia. Beliau memimpin Indonesia dari 1945 hingga 1967. Soekarno dikenal sebagai Bapak Proklamator.",
                        "difficulty": "medium"
                    }
                ]
            },
            Language.SPANISH: {
                "needle_haystack": [
                    {
                        "input": "¿Cuál es la capital de España?",
                        "expected": "Madrid",
                        "context": "España es un país en Europa. Madrid es la capital y ciudad más grande de España. La ciudad es conocida por el Museo del Prado.",
                        "difficulty": "easy"
                    }
                ]
            },
            Language.ARABIC: {
                "needle_haystack": [
                    {
                        "input": "ما هي عاصمة المملكة العربية السعودية؟",
                        "expected": "الرياض",
                        "context": "المملكة العربية السعودية دولة في شبه الجزيرة العربية. الرياض هي العاصمة وأكبر مدينة في المملكة. المدينة تقع في وسط البلاد.",
                        "difficulty": "easy",
                        "rtl_specific": True
                    },
                    {
                        "input": "من هو النبي محمد؟",
                        "expected": "رسول الله وخاتم الأنبياء",
                        "context": "محمد صلى الله عليه وسلم هو رسول الله وخاتم الأنبياء والمرسلين. ولد في مكة المكرمة وبعث بالإسلام.",
                        "difficulty": "medium",
                        "cultural_context": "islamic",
                        "rtl_specific": True
                    }
                ]
            }
        }
        
        # Generate test cases from templates
        templates = test_templates.get(language, {})
        
        for task_type_str, task_tests in templates.items():
            for i, test_data in enumerate(task_tests):
                test_case = TestCase(
                    id=f"{language.value}_{task_type_str}_{i:03d}",
                    language=language.value,
                    task_type=task_type_str,
                    input_text=test_data["input"],
                    expected_output=test_data["expected"],
                    context=test_data["context"],
                    metadata={"generated": True, "template_based": True},
                    difficulty=test_data.get("difficulty", "medium"),
                    cultural_context=test_data.get("cultural_context"),
                    rtl_specific=test_data.get("rtl_specific", False)
                )
                default_tests.append(test_case)
        
        return default_tests
    
    def load_all_languages(self) -> Dict[str, TestSuite]:
        """Load tests for all supported languages"""
        self.logger.info("Loading tests for all languages")
        
        all_suites = {}
        for language in Language:
            suite = self.load_language_tests(language)
            all_suites[language.value] = suite
        
        return all_suites
    
    def get_tests_by_task_type(self, task_type: TaskType, languages: List[Language] = None) -> Dict[str, List[TestCase]]:
        """Get tests filtered by task type across languages"""
        if languages is None:
            languages = list(Language)
        
        filtered_tests = {}
        
        for language in languages:
            if language.value not in self.test_suites:
                self.load_language_tests(language)
            
            suite = self.test_suites[language.value]
            task_tests = [test for test in suite.test_cases if test.task_type == task_type.value]
            
            if task_tests:
                filtered_tests[language.value] = task_tests
        
        return filtered_tests
    
    def get_tests_by_difficulty(self, difficulty: str, languages: List[Language] = None) -> Dict[str, List[TestCase]]:
        """Get tests filtered by difficulty level"""
        if languages is None:
            languages = list(Language)
        
        filtered_tests = {}
        
        for language in languages:
            if language.value not in self.test_suites:
                self.load_language_tests(language)
            
            suite = self.test_suites[language.value]
            difficulty_tests = [test for test in suite.test_cases if test.difficulty == difficulty]
            
            if difficulty_tests:
                filtered_tests[language.value] = difficulty_tests
        
        return filtered_tests
    
    def get_rtl_tests(self) -> List[TestCase]:
        """Get all RTL-specific tests (primarily Arabic)"""
        rtl_tests = []
        
        for language in [Language.ARABIC]:  # Add other RTL languages as needed
            if language.value not in self.test_suites:
                self.load_language_tests(language)
            
            suite = self.test_suites[language.value]
            rtl_tests.extend([test for test in suite.test_cases if test.rtl_specific])
        
        return rtl_tests
    
    def get_cultural_context_tests(self, context: str) -> Dict[str, List[TestCase]]:
        """Get tests filtered by cultural context"""
        context_tests = {}
        
        for language_str, suite in self.test_suites.items():
            cultural_tests = [test for test in suite.test_cases 
                            if test.cultural_context == context]
            
            if cultural_tests:
                context_tests[language_str] = cultural_tests
        
        return context_tests
    
    def export_test_suite(self, language: Language, output_path: str):
        """Export test suite to JSON file"""
        if language.value not in self.test_suites:
            self.load_language_tests(language)
        
        suite = self.test_suites[language.value]
        
        export_data = {
            "language": suite.language,
            "metadata": suite.metadata,
            "total_tests": suite.total_tests,
            "difficulty_distribution": suite.difficulty_distribution,
            "test_cases": []
        }
        
        for test_case in suite.test_cases:
            export_data["test_cases"].append({
                "id": test_case.id,
                "language": test_case.language,
                "task_type": test_case.task_type,
                "input_text": test_case.input_text,
                "expected_output": test_case.expected_output,
                "context": test_case.context,
                "metadata": test_case.metadata,
                "difficulty": test_case.difficulty,
                "cultural_context": test_case.cultural_context,
                "rtl_specific": test_case.rtl_specific
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Exported {suite.total_tests} tests for {language.value} to {output_path}")
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all loaded test suites"""
        summary = {
            "total_languages": len(self.test_suites),
            "languages": list(self.test_suites.keys()),
            "total_tests": sum(suite.total_tests for suite in self.test_suites.values()),
            "tests_by_language": {},
            "tests_by_task_type": {},
            "tests_by_difficulty": {},
            "rtl_tests": len(self.get_rtl_tests()),
            "cultural_contexts": set()
        }
        
        # Aggregate statistics
        for language_str, suite in self.test_suites.items():
            summary["tests_by_language"][language_str] = suite.total_tests
            
            # Aggregate by task type
            for test_case in suite.test_cases:
                task_type = test_case.task_type
                if task_type not in summary["tests_by_task_type"]:
                    summary["tests_by_task_type"][task_type] = 0
                summary["tests_by_task_type"][task_type] += 1
                
                # Aggregate by difficulty
                difficulty = test_case.difficulty
                if difficulty not in summary["tests_by_difficulty"]:
                    summary["tests_by_difficulty"][difficulty] = 0
                summary["tests_by_difficulty"][difficulty] += 1
                
                # Collect cultural contexts
                if test_case.cultural_context:
                    summary["cultural_contexts"].add(test_case.cultural_context)
        
        summary["cultural_contexts"] = list(summary["cultural_contexts"])
        
        return summary

def main():
    """Demo function for test loader"""
    logging.basicConfig(level=logging.INFO)
    
    loader = MultilingualTestLoader()
    
    # Load all languages
    all_suites = loader.load_all_languages()
    
    # Print summary
    summary = loader.get_evaluation_summary()
    print("Multilingual Test Suite Summary:")
    print(f"Total Languages: {summary['total_languages']}")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Languages: {', '.join(summary['languages'])}")
    print(f"RTL Tests: {summary['rtl_tests']}")
    
    print("\nTests by Language:")
    for lang, count in summary['tests_by_language'].items():
        print(f"  {lang}: {count} tests")
    
    print("\nTests by Task Type:")
    for task, count in summary['tests_by_task_type'].items():
        print(f"  {task}: {count} tests")

if __name__ == "__main__":
    main()