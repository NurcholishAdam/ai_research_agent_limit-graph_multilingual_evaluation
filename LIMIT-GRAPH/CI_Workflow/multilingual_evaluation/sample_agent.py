# -*- coding: utf-8 -*-
"""
Sample Multilingual Agent for CI Hook Testing
Demonstrates proper agent interface and multilingual capabilities
"""

import re
from typing import Dict, List, Optional

class MultilingualSampleAgent:
    """
    Sample agent with multilingual capabilities for testing the CI Hook system
    """
    
    def __init__(self):
        """Initialize the sample agent with knowledge bases"""
        
        # Knowledge base for different languages
        self.knowledge_base = {
            # English knowledge
            "en": {
                "capitals": {
                    "france": "Paris",
                    "germany": "Berlin", 
                    "japan": "Tokyo",
                    "italy": "Rome",
                    "spain": "Madrid",
                    "uk": "London",
                    "united kingdom": "London"
                },
                "people": {
                    "einstein": "Albert Einstein was a theoretical physicist who developed the theory of relativity",
                    "shakespeare": "William Shakespeare was an English playwright and poet",
                    "newton": "Isaac Newton was an English mathematician and physicist",
                    "darwin": "Charles Darwin was an English naturalist who proposed the theory of evolution"
                },
                "concepts": {
                    "machine learning": "Machine learning is a subset of artificial intelligence that enables computers to learn from data",
                    "artificial intelligence": "Artificial intelligence is the simulation of human intelligence in machines",
                    "democracy": "Democracy is a form of government where power is held by the people"
                }
            },
            
            # Indonesian knowledge
            "id": {
                "capitals": {
                    "indonesia": "Jakarta",
                    "malaysia": "Kuala Lumpur",
                    "thailand": "Bangkok",
                    "singapura": "Singapura",
                    "filipina": "Manila"
                },
                "people": {
                    "soekarno": "Soekarno adalah presiden pertama Republik Indonesia",
                    "hatta": "Mohammad Hatta adalah wakil presiden pertama Indonesia",
                    "kartini": "Raden Ajeng Kartini adalah pahlawan nasional Indonesia"
                },
                "concepts": {
                    "pancasila": "Pancasila adalah dasar negara Republik Indonesia",
                    "bhinneka tunggal ika": "Bhinneka Tunggal Ika adalah semboyan Indonesia yang berarti berbeda-beda tetapi tetap satu",
                    "gotong royong": "Gotong royong adalah tradisi kerja sama dalam masyarakat Indonesia"
                }
            },
            
            # Arabic knowledge
            "ar": {
                "capitals": {
                    "السعودية": "الرياض",
                    "المملكة العربية السعودية": "الرياض",
                    "مصر": "القاهرة",
                    "الإمارات": "أبوظبي",
                    "الكويت": "الكويت",
                    "قطر": "الدوحة",
                    "الأردن": "عمان"
                },
                "people": {
                    "محمد": "محمد صلى الله عليه وسلم هو رسول الله وخاتم الأنبياء",
                    "عمر": "عمر بن الخطاب هو الخليفة الثاني",
                    "صلاح الدين": "صلاح الدين الأيوبي هو قائد مسلم مشهور"
                },
                "concepts": {
                    "الإسلام": "الإسلام هو دين التوحيد الذي جاء به النبي محمد",
                    "القرآن": "القرآن الكريم هو كتاب الله المنزل على النبي محمد",
                    "الحج": "الحج هو الركن الخامس من أركان الإسلام"
                }
            }
        }
        
        # Common patterns for different question types
        self.question_patterns = {
            "capital": [
                r"capital of (\w+)",
                r"ibu kota (\w+)",
                r"عاصمة (\w+)",
                r"what.*capital.*(\w+)",
                r"apa.*ibu kota.*(\w+)"
            ],
            "who_is": [
                r"who (?:is|was) (\w+)",
                r"siapa (?:itu )?(\w+)",
                r"من (?:هو )?(\w+)"
            ],
            "what_is": [
                r"what is (\w+)",
                r"apa (?:itu )?(\w+)",
                r"ما (?:هو|هي) (\w+)"
            ]
        }
    
    def query(self, question: str, context: str = "") -> str:
        """
        Main query interface for the agent
        
        Args:
            question: Question to answer
            context: Optional context information
            
        Returns:
            Agent's response
        """
        try:
            # Detect language
            language = self._detect_language(question)
            
            # Clean and normalize question
            question_clean = self._normalize_text(question, language)
            
            # Try to find answer in knowledge base
            answer = self._search_knowledge_base(question_clean, language, context)
            
            if answer:
                return answer
            
            # Try pattern matching
            pattern_answer = self._pattern_matching(question_clean, language)
            
            if pattern_answer:
                return pattern_answer
            
            # Use context if available
            if context:
                context_answer = self._extract_from_context(question, context, language)
                if context_answer:
                    return context_answer
            
            # Default response based on language
            return self._default_response(language)
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def _detect_language(self, text: str) -> str:
        """Detect language of input text"""
        # Simple language detection based on character sets and keywords
        
        # Check for Arabic characters
        if re.search(r'[\u0600-\u06FF]', text):
            return "ar"
        
        # Check for Indonesian keywords
        indonesian_keywords = ["apa", "siapa", "dimana", "bagaimana", "mengapa", "ibu kota", "adalah"]
        if any(keyword in text.lower() for keyword in indonesian_keywords):
            return "id"
        
        # Default to English
        return "en"
    
    def _normalize_text(self, text: str, language: str) -> str:
        """Normalize text based on language"""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Language-specific normalization
        if language == "ar":
            # Arabic text normalization
            text = self._normalize_arabic(text)
        
        return text
    
    def _normalize_arabic(self, text: str) -> str:
        """Normalize Arabic text"""
        # Remove diacritics
        diacritics = 'ًٌٍَُِّْ'
        text = ''.join(c for c in text if c not in diacritics)
        
        # Normalize letter variations
        text = re.sub(r'[آأإ]', 'ا', text)  # Alef variations
        text = re.sub(r'ة', 'ه', text)      # Teh Marbuta
        text = re.sub(r'[ىي]', 'ي', text)   # Yeh variations
        
        return text
    
    def _search_knowledge_base(self, question: str, language: str, context: str) -> Optional[str]:
        """Search knowledge base for answer"""
        kb = self.knowledge_base.get(language, {})
        
        # Search in capitals
        for country, capital in kb.get("capitals", {}).items():
            if country in question:
                return capital
        
        # Search in people
        for person, info in kb.get("people", {}).items():
            if person in question:
                return info
        
        # Search in concepts
        for concept, definition in kb.get("concepts", {}).items():
            if concept in question:
                return definition
        
        return None
    
    def _pattern_matching(self, question: str, language: str) -> Optional[str]:
        """Use pattern matching to find answers"""
        
        # Capital city patterns
        for pattern in self.question_patterns["capital"]:
            match = re.search(pattern, question)
            if match:
                country = match.group(1)
                kb = self.knowledge_base.get(language, {})
                return kb.get("capitals", {}).get(country, f"I don't know the capital of {country}")
        
        # Who is patterns
        for pattern in self.question_patterns["who_is"]:
            match = re.search(pattern, question)
            if match:
                person = match.group(1)
                kb = self.knowledge_base.get(language, {})
                return kb.get("people", {}).get(person, f"I don't have information about {person}")
        
        # What is patterns
        for pattern in self.question_patterns["what_is"]:
            match = re.search(pattern, question)
            if match:
                concept = match.group(1)
                kb = self.knowledge_base.get(language, {})
                return kb.get("concepts", {}).get(concept, f"I don't have information about {concept}")
        
        return None
    
    def _extract_from_context(self, question: str, context: str, language: str) -> Optional[str]:
        """Extract answer from provided context"""
        if not context:
            return None
        
        # Simple context extraction based on keyword matching
        question_words = set(question.lower().split())
        context_sentences = context.split('.')
        
        # Find sentences with highest word overlap
        best_sentence = ""
        best_score = 0
        
        for sentence in context_sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words.intersection(sentence_words))
            
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence.strip()
        
        if best_score > 0 and best_sentence:
            return best_sentence
        
        return None
    
    def _default_response(self, language: str) -> str:
        """Return default response based on language"""
        defaults = {
            "en": "I don't have enough information to answer that question accurately.",
            "id": "Saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan tersebut.",
            "ar": "ليس لدي معلومات كافية للإجابة على هذا السؤال بدقة."
        }
        
        return defaults.get(language, defaults["en"])
    
    def get_supported_languages(self) -> List[str]:
        """Return list of supported languages"""
        return list(self.knowledge_base.keys())
    
    def get_capabilities(self) -> Dict[str, List[str]]:
        """Return agent capabilities"""
        return {
            "languages": self.get_supported_languages(),
            "question_types": ["capital_cities", "famous_people", "concepts"],
            "features": ["multilingual", "context_aware", "pattern_matching", "rtl_support"]
        }

# For compatibility with different loading patterns
Agent = MultilingualSampleAgent
agent = MultilingualSampleAgent()

def main():
    """Demo function to test the agent"""
    agent = MultilingualSampleAgent()
    
    # Test questions in different languages
    test_questions = [
        ("What is the capital of France?", ""),
        ("Apa ibu kota Indonesia?", ""),
        ("ما هي عاصمة المملكة العربية السعودية؟", ""),
        ("Who is Einstein?", ""),
        ("Siapa itu Soekarno?", ""),
        ("من هو محمد؟", ""),
        ("What is machine learning?", ""),
        ("Test with context", "The capital of Germany is Berlin. It is a major city in Europe.")
    ]
    
    print("🤖 Multilingual Sample Agent Demo")
    print("=" * 50)
    
    for question, context in test_questions:
        print(f"\nQ: {question}")
        if context:
            print(f"Context: {context}")
        
        response = agent.query(question, context)
        print(f"A: {response}")
    
    print(f"\n📊 Agent Capabilities:")
    capabilities = agent.get_capabilities()
    for category, items in capabilities.items():
        print(f"  {category}: {', '.join(items)}")

if __name__ == "__main__":
    main()