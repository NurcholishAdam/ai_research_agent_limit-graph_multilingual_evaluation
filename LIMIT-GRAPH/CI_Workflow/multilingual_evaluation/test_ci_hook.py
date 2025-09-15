# -*- coding: utf-8 -*-
"""
Test Script for Multilingual Agent Evaluation CI Hook
Validates the complete CI Hook system functionality
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from CI_Hook_Script import CIHookRunner
from load_tests import MultilingualTestLoader, Language
from run_agent import MultilingualAgentRunner
from sample_agent import MultilingualSampleAgent

def setup_test_logging():
    """Setup logging for tests"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_test_loader():
    """Test the multilingual test loader"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing MultilingualTestLoader")
    
    loader = MultilingualTestLoader()
    
    # Test loading all languages
    all_suites = loader.load_all_languages()
    
    assert len(all_suites) > 0, "No test suites loaded"
    assert "en" in all_suites, "English tests not loaded"
    assert "id" in all_suites, "Indonesian tests not loaded"
    assert "ar" in all_suites, "Arabic tests not loaded"
    
    # Test Arabic RTL tests
    rtl_tests = loader.get_rtl_tests()
    assert len(rtl_tests) > 0, "No RTL tests found"
    
    # Test summary generation
    summary = loader.get_evaluation_summary()
    assert summary["total_tests"] > 0, "No tests in summary"
    assert "ar" in summary["languages"], "Arabic not in summary"
    
    logger.info("✅ MultilingualTestLoader tests passed")
    return True

def test_agent_runner():
    """Test the agent runner"""
    logger = logging.getLogger(__name__)
    logger.info("🤖 Testing MultilingualAgentRunner")
    
    # Create test agent
    agent = MultilingualSampleAgent()
    
    # Create runner
    runner = MultilingualAgentRunner()
    
    # Test single test execution
    from load_tests import TestCase
    test_case = TestCase(
        id="test_001",
        language="en",
        task_type="needle_haystack",
        input_text="What is the capital of France?",
        expected_output="Paris",
        context="France is a country in Europe. Paris is the capital of France.",
        metadata={},
        difficulty="easy"
    )
    
    result = runner.run_single_test(agent, test_case)
    assert result.accuracy_score > 0.5, f"Low accuracy score: {result.accuracy_score}"
    assert result.semantic_score > 0.3, f"Low semantic score: {result.semantic_score}"
    
    # Test full evaluation
    metrics = runner.run_evaluation(
        agent,
        languages=[Language.ENGLISH, Language.INDONESIAN, Language.ARABIC],
        max_tests_per_language=3
    )
    
    assert metrics.overall_score > 0.0, "Zero overall score"
    assert metrics.accuracy > 0.0, "Zero accuracy"
    assert "en" in metrics.language_scores, "English not in language scores"
    assert "ar" in metrics.language_scores, "Arabic not in language scores"
    
    logger.info("✅ MultilingualAgentRunner tests passed")
    return True

def test_sample_agent():
    """Test the sample agent"""
    logger = logging.getLogger(__name__)
    logger.info("🎯 Testing Sample Agent")
    
    agent = MultilingualSampleAgent()
    
    # Test English
    response_en = agent.query("What is the capital of France?")
    assert "paris" in response_en.lower(), f"Incorrect English response: {response_en}"
    
    # Test Indonesian
    response_id = agent.query("Apa ibu kota Indonesia?")
    assert "jakarta" in response_id.lower(), f"Incorrect Indonesian response: {response_id}"
    
    # Test Arabic
    response_ar = agent.query("ما هي عاصمة المملكة العربية السعودية؟")
    assert "الرياض" in response_ar, f"Incorrect Arabic response: {response_ar}"
    
    # Test context usage
    context = "Berlin is the capital of Germany and a major European city."
    response_context = agent.query("What is the capital of Germany?", context)
    assert "berlin" in response_context.lower(), f"Context not used: {response_context}"
    
    # Test capabilities
    capabilities = agent.get_capabilities()
    assert "ar" in capabilities["languages"], "Arabic not in capabilities"
    assert "multilingual" in capabilities["features"], "Multilingual not in features"
    
    logger.info("✅ Sample Agent tests passed")
    return True

def test_ci_hook_integration():
    """Test the complete CI Hook integration"""
    logger = logging.getLogger(__name__)
    logger.info("🔗 Testing CI Hook Integration")
    
    # Create temporary agent file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        # Write sample agent to temporary file
        f.write('''
from sample_agent import MultilingualSampleAgent

class TestAgent(MultilingualSampleAgent):
    pass

Agent = TestAgent
agent = TestAgent()
''')
        temp_agent_path = f.name
    
    try:
        # Create CI Hook runner
        ci_hook = CIHookRunner()
        
        # Test agent validation
        validation_result = ci_hook.validate_agent_submission(temp_agent_path, "TestAgent")
        
        assert "agent_name" in validation_result, "Agent name not in validation result"
        assert "validation_passed" in validation_result, "Validation status not in result"
        assert "metrics" in validation_result, "Metrics not in validation result"
        
        # Check if validation passed (should pass with sample agent)
        if validation_result["validation_passed"]:
            logger.info("✅ Agent validation passed")
            
            # Check metrics
            metrics = validation_result["metrics"]
            assert metrics["overall_score"] > 0.0, "Zero overall score in validation"
            assert metrics["accuracy"] > 0.0, "Zero accuracy in validation"
            
        else:
            logger.warning("⚠️ Agent validation failed (expected for basic test)")
            logger.info(f"Failed checks: {validation_result.get('quality_checks', {})}")
        
        logger.info("✅ CI Hook Integration tests passed")
        return True
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_agent_path)
        except:
            pass

def test_configuration():
    """Test configuration loading and validation"""
    logger = logging.getLogger(__name__)
    logger.info("⚙️ Testing Configuration")
    
    # Test default configuration
    ci_hook = CIHookRunner()
    config = ci_hook.config
    
    assert "required_languages" in config, "Required languages not in config"
    assert "quality_thresholds" in config, "Quality thresholds not in config"
    assert "ar" in config["required_languages"], "Arabic not in required languages"
    
    # Test custom configuration
    custom_config = {
        "required_languages": ["en", "ar"],
        "quality_thresholds": {
            "min_overall_score": 0.5,
            "min_accuracy": 0.6
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(custom_config, f)
        temp_config_path = f.name
    
    try:
        ci_hook_custom = CIHookRunner(temp_config_path)
        assert ci_hook_custom.config["quality_thresholds"]["min_overall_score"] == 0.5
        
        logger.info("✅ Configuration tests passed")
        return True
        
    finally:
        try:
            os.unlink(temp_config_path)
        except:
            pass

def test_multilingual_capabilities():
    """Test multilingual capabilities specifically"""
    logger = logging.getLogger(__name__)
    logger.info("🌍 Testing Multilingual Capabilities")
    
    loader = MultilingualTestLoader()
    runner = MultilingualAgentRunner()
    agent = MultilingualSampleAgent()
    
    # Test each language individually
    languages_to_test = [Language.ENGLISH, Language.INDONESIAN, Language.ARABIC]
    
    for language in languages_to_test:
        logger.info(f"Testing {language.value}")
        
        # Load language tests
        test_suite = loader.load_language_tests(language)
        assert test_suite.total_tests > 0, f"No tests for {language.value}"
        
        # Run a few tests
        test_results = []
        for test_case in test_suite.test_cases[:3]:  # Test first 3
            result = runner.run_single_test(agent, test_case)
            test_results.append(result)
        
        # Check that agent responded (even if not perfectly)
        for result in test_results:
            assert len(result.agent_output.strip()) > 0, f"Empty response for {language.value}"
            assert result.error_message is None, f"Error in {language.value}: {result.error_message}"
    
    # Test RTL-specific functionality
    arabic_suite = loader.load_language_tests(Language.ARABIC)
    rtl_tests = [test for test in arabic_suite.test_cases if test.rtl_specific]
    
    if rtl_tests:
        rtl_result = runner.run_single_test(agent, rtl_tests[0])
        assert len(rtl_result.agent_output.strip()) > 0, "Empty RTL response"
        logger.info("✅ RTL processing test passed")
    
    logger.info("✅ Multilingual capabilities tests passed")
    return True

def run_all_tests():
    """Run all tests"""
    logger = setup_test_logging()
    
    logger.info("🚀 Starting CI Hook System Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Test Loader", test_test_loader),
        ("Agent Runner", test_agent_runner),
        ("Sample Agent", test_sample_agent),
        ("Configuration", test_configuration),
        ("Multilingual Capabilities", test_multilingual_capabilities),
        ("CI Hook Integration", test_ci_hook_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n--- Running {test_name} ---")
            result = test_func()
            if result:
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                failed += 1
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"💥 {test_name} CRASHED: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("🏁 Test Results Summary")
    logger.info("=" * 60)
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        logger.info("🎉 All tests passed! CI Hook system is ready.")
        return True
    else:
        logger.error("⚠️ Some tests failed. Please review and fix issues.")
        return False

def main():
    """Main test function"""
    success = run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()