# Multilingual Agent Evaluation CI Hook

## Overview

The Multilingual Agent Evaluation CI Hook provides automated validation of agent submissions across multiple languages with comprehensive diagnostic capabilities. This system evaluates agents on English, Indonesian, and Arabic languages with specialized RTL text processing and cultural context awareness.

## 🌍 Supported Languages

- **English (EN)**: Full evaluation with academic and technical contexts
- **Indonesian (ID)**: Southeast Asian cultural context with Islamic references
- **Arabic (AR)**: RTL text processing with Middle Eastern cultural awareness

## 🔬 Evaluation Dimensions

### Core Metrics
- **Accuracy**: Exact and partial match scoring
- **Semantic Similarity**: TF-IDF and cosine similarity analysis
- **Response Time**: Performance and efficiency measurement
- **Error Rate**: Robustness and reliability assessment

### Specialized Metrics
- **Distractor Robustness**: Resistance to misleading information
- **Context Length Handling**: Performance across varying input lengths
- **RTL Text Processing**: Arabic right-to-left text handling
- **Cultural Awareness**: Context-appropriate responses
- **Cross-lingual Consistency**: Performance stability across languages

## 📁 Components

### 1. Test Loader (`load_tests.py`)
- Manages multilingual test suites
- Supports multiple task types and difficulty levels
- Generates default tests for each language
- Handles RTL-specific test cases

### 2. Agent Runner (`run_agent.py`)
- Executes agents against test suites
- Calculates comprehensive metrics
- Supports multiple agent interfaces
- Generates detailed evaluation reports

### 3. CI Hook Script (`CI_Hook_Script.py`)
- Main orchestration script for CI pipeline
- Validates agent submissions
- Performs quality checks against thresholds
- Updates leaderboard and generates reports

### 4. GitHub Actions Workflow (`GitHub_Actions_Workflow.yml`)
- Automated CI/CD pipeline
- Detects changed agent files
- Runs parallel evaluations
- Posts results to PRs and updates artifacts

## 🚀 Usage

### Manual Evaluation
```bash
# Evaluate a single agent
python CI_Hook_Script.py path/to/agent.py --agent-name MyAgent

# With custom configuration
python CI_Hook_Script.py path/to/agent.py --config custom_config.json

# Enable debug logging
python CI_Hook_Script.py path/to/agent.py --debug
```

### GitHub Actions Integration

#### Automatic Triggers
- **Pull Requests**: Evaluates changed agent files in `agents/` or `submissions/`
- **Push to Main**: Re-evaluates agents and updates leaderboard
- **Manual Dispatch**: Evaluate specific agent with custom parameters

#### Manual Trigger
```yaml
# In GitHub Actions UI
workflow_dispatch:
  inputs:
    agent_path: "agents/my_agent.py"
    agent_name: "MyAgent"
    languages: "en,id,ar"
```

## 📊 Quality Thresholds

### Default Thresholds
- **Minimum Overall Score**: 0.6
- **Minimum Accuracy**: 0.7
- **Minimum Semantic Score**: 0.6
- **Maximum Error Rate**: 20%
- **Minimum RTL Score**: 0.5
- **Minimum Cultural Awareness**: 0.5

### Customization
Edit `ci_config.json` to adjust thresholds:
```json
{
  "quality_thresholds": {
    "min_overall_score": 0.65,
    "min_accuracy": 0.75,
    "min_semantic_score": 0.65,
    "max_error_rate": 0.15
  }
}
```

## 🏗️ Agent Requirements

### Interface Requirements
Agents must implement one of the following interfaces:

#### Option 1: Query Interface
```python
class MyAgent:
    def query(self, question: str, context: str = "") -> str:
        # Process question with optional context
        return "Agent response"
```

#### Option 2: Generate Interface
```python
class MyAgent:
    def generate(self, prompt: str, context: str = "") -> str:
        # Generate response for prompt
        return "Generated response"
```

#### Option 3: Callable Interface
```python
class MyAgent:
    def __call__(self, question: str, context: str = "") -> str:
        # Direct callable interface
        return "Agent response"
```

### File Structure
```
agents/
├── my_agent.py          # Agent implementation
└── requirements.txt     # Optional dependencies

submissions/
├── team_agent.py        # Submission from external team
└── README.md           # Submission documentation
```

## 📈 Evaluation Process

### 1. Agent Detection
- Scans for changed Python files in `agents/` and `submissions/`
- Validates file existence and basic syntax
- Extracts agent name from filename

### 2. Test Loading
- Loads language-specific test suites
- Generates default tests if none exist
- Filters by task type and difficulty

### 3. Agent Execution
- Runs agent against each test case
- Measures response time and accuracy
- Handles errors gracefully

### 4. Metric Calculation
- Computes core and specialized metrics
- Aggregates scores by language and task type
- Calculates overall weighted score

### 5. Quality Validation
- Checks metrics against thresholds
- Validates language coverage
- Assesses performance requirements

### 6. Reporting
- Generates detailed evaluation reports
- Updates leaderboard for passing agents
- Posts results to GitHub PRs

## 📋 Test Types

### Needle-in-Haystack
- Tests information retrieval from context
- Varies context length and complexity
- Includes distractor information

### Question Answering
- Direct question-answer pairs
- Cultural context awareness
- Domain-specific knowledge

### Semantic Similarity
- Evaluates understanding of meaning
- Cross-lingual concept matching
- Paraphrase recognition

### Distractor Resistance
- Tests robustness to misleading information
- Semantic and syntactic distractors
- Adversarial content injection

### Context Length Handling
- Evaluates performance across input lengths
- Tests from 100 to 4000+ tokens
- Measures degradation patterns

### RTL Processing (Arabic)
- Right-to-left text handling
- Bidirectional text compliance
- Arabic script variations

### Cultural Awareness
- Context-appropriate responses
- Cultural sensitivity assessment
- Regional knowledge evaluation

## 🏆 Leaderboard

### Ranking Criteria
Agents are ranked by overall score, calculated as:
```
Overall Score = 0.25×Accuracy + 0.20×Semantic + 0.15×Distractor + 
                0.15×Context + 0.10×RTL + 0.10×Cultural + 0.05×Consistency
```

### Leaderboard Format
```json
{
  "agent_name": "TopAgent",
  "overall_score": 0.847,
  "accuracy": 0.892,
  "semantic_score": 0.876,
  "rtl_handling_score": 0.823,
  "cultural_awareness_score": 0.798,
  "language_scores": {
    "en": 0.895,
    "id": 0.834,
    "ar": 0.756
  }
}
```

## 🔧 Configuration

### CI Configuration (`ci_config.json`)
```json
{
  "required_languages": ["en", "id", "ar"],
  "max_evaluation_time": 1800,
  "quality_thresholds": {
    "min_overall_score": 0.6,
    "min_accuracy": 0.7
  },
  "max_tests_per_language": 10,
  "enable_leaderboard_update": true
}
```

### Environment Variables
```bash
# GitHub Actions
GITHUB_TOKEN=<token>           # For PR comments
GITHUB_STEP_SUMMARY=<file>     # For job summaries

# Evaluation
CI_DEBUG=true                  # Enable debug logging
LANGUAGES=en,id,ar            # Languages to test
MIN_OVERALL_SCORE=0.6         # Quality threshold
```

## 📊 Outputs

### Evaluation Results
- **JSON Export**: Detailed metrics and test results
- **Markdown Report**: Human-readable evaluation summary
- **GitHub Summary**: CI job summary with key metrics
- **PR Comments**: Automated feedback on submissions

### Artifacts
- `evaluation_results/`: Detailed evaluation data
- `ci_evaluation.log`: Execution logs
- `leaderboard.json`: Updated rankings
- Agent-specific reports and metrics

## 🐛 Troubleshooting

### Common Issues

#### Agent Loading Errors
```bash
# Check agent file syntax
python -m py_compile agents/my_agent.py

# Verify agent interface
python -c "
import sys
sys.path.append('agents')
from my_agent import MyAgent
agent = MyAgent()
print(hasattr(agent, 'query'))
"
```

#### Evaluation Timeouts
- Reduce `max_tests_per_language` in config
- Optimize agent response time
- Check for infinite loops or blocking operations

#### Low Scores
- Review failed test cases in detailed results
- Check language-specific performance
- Analyze error messages and traces

#### RTL Processing Issues
- Ensure proper UTF-8 encoding
- Test with Arabic text samples
- Verify bidirectional text handling

### Debug Mode
```bash
# Enable detailed logging
python CI_Hook_Script.py agents/my_agent.py --debug

# Check specific language performance
python -c "
from load_tests import MultilingualTestLoader, Language
loader = MultilingualTestLoader()
suite = loader.load_language_tests(Language.ARABIC)
print(f'Arabic tests: {len(suite.test_cases)}')
"
```

## 🤝 Contributing

### Adding New Languages
1. Update `Language` enum in `load_tests.py`
2. Add language configuration in `ci_config.json`
3. Create test templates for the new language
4. Update quality thresholds if needed

### Adding New Task Types
1. Update `TaskType` enum in `load_tests.py`
2. Implement task-specific evaluation logic
3. Add test cases for the new task type
4. Update metric calculation weights

### Improving Evaluation
1. Enhance semantic similarity calculation
2. Add more sophisticated cultural awareness tests
3. Implement additional robustness measures
4. Optimize performance and scalability

## 📄 License

This CI Hook system is part of the LIMIT-GRAPH project and follows the same licensing terms.