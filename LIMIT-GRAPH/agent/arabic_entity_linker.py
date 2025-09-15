# -*- coding: utf-8 -*-
"""
Arabic Entity Linker for LIMIT-GRAPH
Specialized entity linking for Arabic language processing with RTL support
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging
import unicodedata

class ArabicEntityLinker:
    """
    Arabic-specific entity linker with cultural and linguistic awareness
    """
    
    def __init__(self):
        """Initialize Arabic entity linker"""
        self.logger = logging.getLogger(__name__)
        
        # Arabic-specific entity patterns
        self.entity_patterns = {
            'person': [
                r'أحمد|محمد|علي|فاطمة|عائشة|سارة|خالد|نورا|يوسف|مريم',
                r'الدكتور|الأستاذ|المهندس|الطبيب|المعلم',
                r'الرئيس|الوزير|المدير|القائد'
            ],
            'location': [
                r'الرياض|جدة|مكة|المدينة|الدمام|الطائف|أبها',
                r'المسجد|المدرسة|المستشفى|الجامعة|المكتب|البيت',
                r'السعودية|مصر|الإمارات|الكويت|قطر|البحرين'
            ],
            'organization': [
                r'الشركة|المؤسسة|الوزارة|الجامعة|المستشفى',
                r'أرامكو|سابك|الراجحي|البنك_الأهلي',
                r'الحكومة|البرلمان|المجلس'
            ],
            'object': [
                r'السيارة|المنزل|الكتاب|الحاسوب|الهاتف',
                r'المشروع|البرنامج|النظام|التطبيق',
                r'الطعام|الماء|الدواء|الملابس'
            ]
        }
        
        # Arabic relation patterns
        self.relation_patterns = {
            'ownership': ['يملك', 'لديه', 'يحتوي_على', 'يضم'],
            'location': ['يقع_في', 'موجود_في', 'يتواجد_في', 'في'],
            'work': ['يعمل_في', 'يعمل_لدى', 'موظف_في', 'يشتغل_في'],
            'family': ['والد', 'أم', 'ابن', 'ابنة', 'أخ', 'أخت', 'زوج', 'زوجة'],
            'social': ['صديق', 'زميل', 'جار', 'شريك', 'معارف'],
            'management': ['يدير', 'يشرف_على', 'يقود', 'رئيس', 'مدير'],
            'action': ['يفعل', 'يقوم_بـ', 'ينجز', 'يعمل_على', 'يطور']
        }
        
        # Arabic linguistic features
        self.arabic_features = {
            'definite_article': 'ال',
            'rtl_direction': True,
            'diacritics': 'ًٌٍَُِّْ',
            'common_prefixes': ['و', 'ف', 'ب', 'ل', 'ك'],
            'common_suffixes': ['ة', 'ان', 'ين', 'ون', 'ها', 'هم', 'هن']
        }
        
        # Initialize Arabic NLP components
        self._initialize_arabic_nlp()
    
    def _initialize_arabic_nlp(self):
        """Initialize Arabic NLP processing components"""
        try:
            # Try to import Arabic NLP libraries
            import pyarabic.araby as araby
            self.araby = araby
            self.arabic_nlp_available = True
            self.logger.info("Arabic NLP components initialized successfully")
        except ImportError:
            self.logger.warning("Arabic NLP libraries not available, using basic processing")
            self.araby = None
            self.arabic_nlp_available = False
    
    def get_entity_info(self, entity: str) -> Dict[str, Any]:
        """
        Get information about an Arabic entity
        
        Args:
            entity: Arabic entity text
            
        Returns:
            Dictionary with entity information
        """
        entity_info = {
            'text': entity,
            'normalized': self._normalize_arabic_text(entity),
            'type': self._classify_entity_type(entity),
            'features': self._extract_arabic_features(entity),
            'alternatives': self._generate_entity_alternatives(entity)
        }
        
        return entity_info
    
    def _normalize_arabic_text(self, text: str) -> str:
        """Normalize Arabic text"""
        if not text:
            return ""
        
        # Remove diacritics
        normalized = self._remove_diacritics(text)
        
        # Normalize Arabic letters
        normalized = self._normalize_arabic_letters(normalized)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _remove_diacritics(self, text: str) -> str:
        """Remove Arabic diacritics"""
        if self.arabic_nlp_available and self.araby:
            return self.araby.strip_diacritics(text)
        else:
            # Simple diacritic removal
            diacritics = 'ًٌٍَُِّْ'
            return ''.join(c for c in text if c not in diacritics)
    
    def _normalize_arabic_letters(self, text: str) -> str:
        """Normalize Arabic letter variations"""
        # Normalize Alef variations
        text = re.sub(r'[آأإ]', 'ا', text)
        
        # Normalize Teh Marbuta
        text = re.sub(r'ة', 'ه', text)
        
        # Normalize Yeh variations
        text = re.sub(r'[ىي]', 'ي', text)
        
        return text
    
    def _classify_entity_type(self, entity: str) -> str:
        """Classify Arabic entity type"""
        normalized_entity = self._normalize_arabic_text(entity).lower()
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, normalized_entity):
                    return entity_type
        
        # Default classification based on linguistic features
        if self._has_definite_article(entity):
            if self._is_proper_noun_pattern(entity):
                return 'person'
            else:
                return 'object'
        
        return 'unknown'
    
    def _extract_arabic_features(self, entity: str) -> Dict[str, Any]:
        """Extract Arabic linguistic features"""
        features = {
            'has_definite_article': self._has_definite_article(entity),
            'is_proper_noun': self._is_proper_noun_pattern(entity),
            'length': len(entity),
            'word_count': len(entity.split()),
            'has_diacritics': self._has_diacritics(entity),
            'rtl_text': True,  # All Arabic text is RTL
            'script': 'arabic'
        }
        
        return features
    
    def _generate_entity_alternatives(self, entity: str) -> List[str]:
        """Generate alternative forms of Arabic entity"""
        alternatives = [entity]
        
        # Add normalized version
        normalized = self._normalize_arabic_text(entity)
        if normalized != entity:
            alternatives.append(normalized)
        
        # Add version without definite article
        if self._has_definite_article(entity):
            without_al = self._remove_definite_article(entity)
            alternatives.append(without_al)
        
        # Add version with definite article
        if not self._has_definite_article(entity):
            with_al = 'ال' + entity
            alternatives.append(with_al)
        
        # Add diacritic variations if available
        if self.arabic_nlp_available:
            alternatives.extend(self._generate_diacritic_variations(entity))
        
        return list(set(alternatives))  # Remove duplicates
    
    def _has_definite_article(self, text: str) -> bool:
        """Check if text has Arabic definite article"""
        return text.startswith('ال')
    
    def _remove_definite_article(self, text: str) -> str:
        """Remove Arabic definite article"""
        if self._has_definite_article(text):
            return text[2:]  # Remove 'ال'
        return text
    
    def _is_proper_noun_pattern(self, entity: str) -> bool:
        """Check if entity follows proper noun patterns"""
        # Simple heuristic: proper nouns often don't have definite articles
        # or are in specific patterns
        if not self._has_definite_article(entity):
            return True
        
        # Check against known proper noun patterns
        proper_patterns = [
            r'^(أحمد|محمد|علي|فاطمة|عائشة|سارة|خالد|نورا|يوسف|مريم)',
            r'^(الرياض|جدة|مكة|المدينة|الدمام)',
            r'^(السعودية|مصر|الإمارات|الكويت)'
        ]
        
        for pattern in proper_patterns:
            if re.match(pattern, entity):
                return True
        
        return False
    
    def _has_diacritics(self, text: str) -> bool:
        """Check if text contains Arabic diacritics"""
        diacritics = 'ًٌٍَُِّْ'
        return any(c in diacritics for c in text)
    
    def _generate_diacritic_variations(self, entity: str) -> List[str]:
        """Generate diacritic variations of entity"""
        variations = []
        
        # This is a simplified approach
        # In a real implementation, you'd use proper Arabic morphology
        
        # Add common diacritic patterns
        base = self._remove_diacritics(entity)
        
        # Common patterns for different word types
        if self._is_proper_noun_pattern(entity):
            # Proper nouns often have specific diacritic patterns
            variations.extend([
                base + 'ُ',  # Nominative
                base + 'َ',  # Accusative
                base + 'ِ'   # Genitive
            ])
        
        return variations
    
    def link_entities_in_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Link entities in Arabic graph with enhanced information
        
        Args:
            graph: Graph structure with Arabic entities
            
        Returns:
            Enhanced graph with entity linking information
        """
        enhanced_graph = graph.copy()
        
        # Process nodes
        if 'nodes' in graph:
            enhanced_nodes = []
            for node in graph['nodes']:
                if isinstance(node, str):
                    entity_info = self.get_entity_info(node)
                    enhanced_node = {
                        'id': node,
                        'entity_info': entity_info
                    }
                    enhanced_nodes.append(enhanced_node)
                else:
                    enhanced_nodes.append(node)
            
            enhanced_graph['enhanced_nodes'] = enhanced_nodes
        
        # Process edges with Arabic relation classification
        if 'edges' in graph:
            enhanced_edges = []
            for edge in graph['edges']:
                enhanced_edge = edge.copy()
                
                # Classify relation type
                relation = edge.get('relation', '')
                if relation and relation != '[MASK]':
                    relation_type = self._classify_relation_type(relation)
                    enhanced_edge['relation_type'] = relation_type
                    enhanced_edge['relation_alternatives'] = self._get_relation_alternatives(relation)
                
                # Enhance source and target entities
                if 'source' in edge:
                    enhanced_edge['source_info'] = self.get_entity_info(edge['source'])
                
                if 'target' in edge:
                    enhanced_edge['target_info'] = self.get_entity_info(edge['target'])
                
                enhanced_edges.append(enhanced_edge)
            
            enhanced_graph['enhanced_edges'] = enhanced_edges
        
        return enhanced_graph
    
    def _classify_relation_type(self, relation: str) -> str:
        """Classify Arabic relation type"""
        normalized_relation = self._normalize_arabic_text(relation).lower()
        
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                if pattern in normalized_relation or normalized_relation in pattern:
                    return relation_type
        
        return 'unknown'
    
    def _get_relation_alternatives(self, relation: str) -> List[str]:
        """Get alternative forms of Arabic relation"""
        alternatives = [relation]
        
        # Add normalized version
        normalized = self._normalize_arabic_text(relation)
        if normalized != relation:
            alternatives.append(normalized)
        
        # Find similar relations in the same category
        relation_type = self._classify_relation_type(relation)
        if relation_type in self.relation_patterns:
            similar_relations = self.relation_patterns[relation_type]
            alternatives.extend(similar_relations)
        
        return list(set(alternatives))
    
    def suggest_masked_relations(self, source_entity: str, target_entity: str, 
                               context: str = "") -> List[Dict[str, Any]]:
        """
        Suggest possible relations for masked Arabic edges
        
        Args:
            source_entity: Source entity in Arabic
            target_entity: Target entity in Arabic
            context: Additional context
            
        Returns:
            List of suggested relations with confidence scores
        """
        suggestions = []
        
        # Get entity information
        source_info = self.get_entity_info(source_entity)
        target_info = self.get_entity_info(target_entity)
        
        source_type = source_info['type']
        target_type = target_info['type']
        
        # Suggest relations based on entity types
        type_based_suggestions = self._get_type_based_relations(source_type, target_type)
        
        for relation, confidence in type_based_suggestions:
            suggestion = {
                'relation': relation,
                'confidence': confidence,
                'reasoning': f"Based on entity types: {source_type} -> {target_type}",
                'source_type': source_type,
                'target_type': target_type
            }
            suggestions.append(suggestion)
        
        # Context-based suggestions
        if context:
            context_suggestions = self._get_context_based_relations(context, source_entity, target_entity)
            suggestions.extend(context_suggestions)
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _get_type_based_relations(self, source_type: str, target_type: str) -> List[Tuple[str, float]]:
        """Get relation suggestions based on entity types"""
        type_relations = {
            ('person', 'person'): [('صديق', 0.7), ('زميل', 0.6), ('أخ', 0.5), ('والد', 0.5)],
            ('person', 'location'): [('يسكن_في', 0.8), ('يعمل_في', 0.7), ('يزور', 0.5)],
            ('person', 'object'): [('يملك', 0.8), ('يستخدم', 0.6), ('يحب', 0.4)],
            ('person', 'organization'): [('يعمل_في', 0.9), ('يدير', 0.6), ('عضو_في', 0.5)],
            ('organization', 'location'): [('يقع_في', 0.9), ('فرع_في', 0.6)],
            ('object', 'location'): [('موجود_في', 0.8), ('يقع_في', 0.7)]
        }
        
        return type_relations.get((source_type, target_type), [('مرتبط_بـ', 0.3)])
    
    def _get_context_based_relations(self, context: str, source: str, target: str) -> List[Dict[str, Any]]:
        """Get relation suggestions based on context"""
        suggestions = []
        
        context_lower = context.lower()
        
        # Look for relation keywords in context
        for relation_type, relations in self.relation_patterns.items():
            for relation in relations:
                if relation in context_lower:
                    suggestion = {
                        'relation': relation,
                        'confidence': 0.6,
                        'reasoning': f"Found '{relation}' in context",
                        'source_type': 'context',
                        'target_type': 'context'
                    }
                    suggestions.append(suggestion)
        
        return suggestions
    
    def evaluate_rtl_handling(self, text: str) -> Dict[str, Any]:
        """
        Evaluate RTL (Right-to-Left) text handling capabilities
        
        Args:
            text: Arabic text to evaluate
            
        Returns:
            RTL evaluation metrics
        """
        evaluation = {
            'is_rtl': True,
            'text_direction': 'rtl',
            'has_mixed_direction': self._has_mixed_direction(text),
            'rtl_markers': self._find_rtl_markers(text),
            'bidi_compliance': self._check_bidi_compliance(text),
            'rendering_issues': self._detect_rendering_issues(text)
        }
        
        return evaluation
    
    def _has_mixed_direction(self, text: str) -> bool:
        """Check if text has mixed RTL/LTR content"""
        # Simple check for Latin characters mixed with Arabic
        has_arabic = bool(re.search(r'[\u0600-\u06FF]', text))
        has_latin = bool(re.search(r'[a-zA-Z]', text))
        
        return has_arabic and has_latin
    
    def _find_rtl_markers(self, text: str) -> List[str]:
        """Find RTL direction markers in text"""
        markers = []
        
        # Unicode RTL markers
        rtl_markers = {
            '\u202E': 'RLO (Right-to-Left Override)',
            '\u202D': 'LRO (Left-to-Right Override)', 
            '\u202C': 'PDF (Pop Directional Formatting)',
            '\u061C': 'ALM (Arabic Letter Mark)'
        }
        
        for marker, description in rtl_markers.items():
            if marker in text:
                markers.append(description)
        
        return markers
    
    def _check_bidi_compliance(self, text: str) -> bool:
        """Check if text follows bidirectional text standards"""
        # Simplified check - in real implementation, use proper bidi algorithm
        
        # Check for proper RTL/LTR handling
        if self._has_mixed_direction(text):
            # Should have proper directional markers
            return len(self._find_rtl_markers(text)) > 0
        
        return True  # Pure Arabic text is generally compliant
    
    def _detect_rendering_issues(self, text: str) -> List[str]:
        """Detect potential Arabic text rendering issues"""
        issues = []
        
        # Check for isolated Arabic letters (should be connected)
        if re.search(r'[\u0600-\u06FF]\s+[\u0600-\u06FF]', text):
            issues.append("Potential letter connection issues")
        
        # Check for improper diacritic placement
        if re.search(r'[\u064B-\u065F]{2,}', text):
            issues.append("Multiple consecutive diacritics")
        
        # Check for mixed number directions
        if re.search(r'[\u0660-\u0669].*[0-9]|[0-9].*[\u0660-\u0669]', text):
            issues.append("Mixed Arabic and Western numerals")
        
        return issues
        self.logger = logging.getLogger(__name__)
        
        # Arabic language patterns
        self.arabic_patterns = {
            # Person names patterns
            "person_prefixes": ["أبو", "أم", "ابن", "بنت", "الشيخ", "الدكتور", "الأستاذ"],
            "person_suffixes": ["الدين", "الله", "الرحمن", "الملك"],
            
            # Location patterns
            "location_prefixes": ["مدينة", "قرية", "حي", "شارع", "جامعة", "مسجد", "مدرسة"],
            "location_suffixes": ["آباد", "ستان", "ية"],
            
            # Organization patterns
            "org_prefixes": ["شركة", "مؤسسة", "جمعية", "منظمة", "حزب", "وزارة"],
            "org_suffixes": ["المحدودة", "والشركاه", "للتجارة"],
            
            # Relationship patterns
            "family_relations": ["أب", "أم", "ابن", "بنت", "أخ", "أخت", "جد", "جدة", "عم", "عمة", "خال", "خالة"],
            "social_relations": ["صديق", "زميل", "جار", "معلم", "طالب", "زوج", "زوجة"],
            "professional_relations": ["مدير", "موظف", "عامل", "طبيب", "مهندس", "محامي"]
        }
        
        # Arabic diacritics for normalization
        self.arabic_diacritics = [
            '\u064B', '\u064C', '\u064D', '\u064E', '\u064F', '\u0650',
            '\u0651', '\u0652', '\u0653', '\u0654', '\u0655', '\u0656',
            '\u0657', '\u0658', '\u0659', '\u065A', '\u065B', '\u065C',
            '\u065D', '\u065E', '\u065F', '\u0670'
        ]
        
        # Common Arabic relations for graph construction
        self.arabic_relations = {
            # Ownership/Possession
            "يملك": {"type": "ownership", "strength": 0.9, "bidirectional": False},
            "لديه": {"type": "possession", "strength": 0.8, "bidirectional": False},
            "يحتوي": {"type": "containment", "strength": 0.7, "bidirectional": False},
            
            # Location
            "في": {"type": "location", "strength": 0.9, "bidirectional": False},
            "يعيش في": {"type": "residence", "strength": 0.9, "bidirectional": False},
            "يقع في": {"type": "location", "strength": 0.8, "bidirectional": False},
            "بجانب": {"type": "proximity", "strength": 0.6, "bidirectional": True},
            
            # Family Relations
            "والد": {"type": "family", "strength": 1.0, "bidirectional": False},
            "والدة": {"type": "family", "strength": 1.0, "bidirectional": False},
            "ابن": {"type": "family", "strength": 1.0, "bidirectional": False},
            "ابنة": {"type": "family", "strength": 1.0, "bidirectional": False},
            "أخ": {"type": "family", "strength": 0.9, "bidirectional": True},
            "أخت": {"type": "family", "strength": 0.9, "bidirectional": True},
            "زوج": {"type": "family", "strength": 1.0, "bidirectional": True},
            "زوجة": {"type": "family", "strength": 1.0, "bidirectional": True},
            
            # Social Relations
            "صديق": {"type": "social", "strength": 0.7, "bidirectional": True},
            "زميل": {"type": "professional", "strength": 0.6, "bidirectional": True},
            "جار": {"type": "social", "strength": 0.5, "bidirectional": True},
            
            # Professional Relations
            "يعمل في": {"type": "employment", "strength": 0.8, "bidirectional": False},
            "مدير": {"type": "management", "strength": 0.9, "bidirectional": False},
            "موظف": {"type": "employment", "strength": 0.7, "bidirectional": False},
            "يدرس في": {"type": "education", "strength": 0.8, "bidirectional": False},
            "معلم": {"type": "education", "strength": 0.8, "bidirectional": False},
            
            # Actions
            "يقرأ": {"type": "action", "strength": 0.6, "bidirectional": False},
            "يكتب": {"type": "action", "strength": 0.6, "bidirectional": False},
            "يلعب": {"type": "action", "strength": 0.5, "bidirectional": False},
            "يدرس": {"type": "action", "strength": 0.7, "bidirectional": False}
        }
        
        # Entity type indicators
        self.entity_type_indicators = {
            "person": {
                "patterns": [r"محمد|أحمد|علي|فاطمة|عائشة|خديجة", r"أبو\s+\w+", r"بن\s+\w+"],
                "prefixes": ["الأستاذ", "الدكتور", "المهندس", "الشيخ"],
                "suffixes": ["الدين", "الله"]
            },
            "location": {
                "patterns": [r"مكة|المدينة|الرياض|القاهرة|بغداد|دمشق", r"مدينة\s+\w+", r"شارع\s+\w+"],
                "prefixes": ["مدينة", "قرية", "حي", "شارع", "جامعة"],
                "suffixes": ["آباد", "ستان"]
            },
            "organization": {
                "patterns": [r"شركة\s+\w+", r"جامعة\s+\w+", r"وزارة\s+\w+"],
                "prefixes": ["شركة", "مؤسسة", "جمعية", "منظمة", "وزارة"],
                "suffixes": ["المحدودة", "والشركاه"]
            },
            "object": {
                "patterns": [r"كتاب|سيارة|بيت|مكتب|هاتف"],
                "prefixes": [],
                "suffixes": []
            }
        }
        
        # Initialize entity knowledge base
        self.entity_kb = self._initialize_entity_kb()
    
    def _initialize_entity_kb(self) -> Dict[str, Dict[str, Any]]:
        """Initialize Arabic entity knowledge base"""
        kb = {
            # Common Arabic names
            "محمد": {"type": "person", "gender": "male", "frequency": 0.9},
            "أحمد": {"type": "person", "gender": "male", "frequency": 0.8},
            "علي": {"type": "person", "gender": "male", "frequency": 0.8},
            "فاطمة": {"type": "person", "gender": "female", "frequency": 0.7},
            "عائشة": {"type": "person", "gender": "female", "frequency": 0.6},
            "خديجة": {"type": "person", "gender": "female", "frequency": 0.6},
            
            # Common locations
            "مكة": {"type": "location", "country": "السعودية", "significance": "religious"},
            "المدينة": {"type": "location", "country": "السعودية", "significance": "religious"},
            "الرياض": {"type": "location", "country": "السعودية", "significance": "capital"},
            "القاهرة": {"type": "location", "country": "مصر", "significance": "capital"},
            "بغداد": {"type": "location", "country": "العراق", "significance": "capital"},
            "دمشق": {"type": "location", "country": "سوريا", "significance": "capital"},
            
            # Common objects
            "كتاب": {"type": "object", "category": "literature", "frequency": 0.8},
            "سيارة": {"type": "object", "category": "vehicle", "frequency": 0.7},
            "بيت": {"type": "object", "category": "building", "frequency": 0.9},
            "مكتب": {"type": "object", "category": "furniture", "frequency": 0.6},
            "هاتف": {"type": "object", "category": "technology", "frequency": 0.8},
            
            # Organizations
            "جامعة الملك سعود": {"type": "organization", "category": "education", "country": "السعودية"},
            "الأزهر": {"type": "organization", "category": "religious", "country": "مصر"}
        }
        
        return kb
    
    def normalize_arabic_text(self, text: str) -> str:
        """Normalize Arabic text by removing diacritics and standardizing characters"""
        if not text:
            return ""
        
        # Remove diacritics
        normalized = text
        for diacritic in self.arabic_diacritics:
            normalized = normalized.replace(diacritic, '')
        
        # Normalize Arabic characters
        normalized = normalized.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        normalized = normalized.replace('ة', 'ه')
        normalized = normalized.replace('ى', 'ي')
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def detect_entity_type(self, entity: str) -> Tuple[str, float]:
        """
        Detect entity type for Arabic entity
        
        Args:
            entity: Arabic entity text
            
        Returns:
            Tuple of (entity_type, confidence_score)
        """
        entity = self.normalize_arabic_text(entity)
        
        # Check knowledge base first
        if entity in self.entity_kb:
            return self.entity_kb[entity]["type"], 0.9
        
        # Pattern-based detection
        for entity_type, indicators in self.entity_type_indicators.items():
            confidence = 0.0
            
            # Check patterns
            for pattern in indicators["patterns"]:
                if re.search(pattern, entity):
                    confidence = max(confidence, 0.8)
            
            # Check prefixes
            for prefix in indicators["prefixes"]:
                if entity.startswith(prefix):
                    confidence = max(confidence, 0.7)
            
            # Check suffixes
            for suffix in indicators["suffixes"]:
                if entity.endswith(suffix):
                    confidence = max(confidence, 0.6)
            
            if confidence > 0.5:
                return entity_type, confidence
        
        # Default classification
        return "unknown", 0.3
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract Arabic entities from text
        
        Args:
            text: Arabic text to process
            
        Returns:
            List of extracted entities with metadata
        """
        text = self.normalize_arabic_text(text)
        entities = []
        
        # Split text into potential entities
        words = text.split()
        
        # Single word entities
        for i, word in enumerate(words):
            if self._is_potential_entity(word):
                entity_type, confidence = self.detect_entity_type(word)
                
                entity = {
                    "text": word,
                    "type": entity_type,
                    "confidence": confidence,
                    "position": i,
                    "length": 1,
                    "normalized": self.normalize_arabic_text(word)
                }
                
                entities.append(entity)
        
        # Multi-word entities (2-3 words)
        for i in range(len(words) - 1):
            for j in range(2, min(4, len(words) - i + 1)):
                phrase = ' '.join(words[i:i+j])
                
                if self._is_potential_entity(phrase):
                    entity_type, confidence = self.detect_entity_type(phrase)
                    
                    if confidence > 0.6:  # Higher threshold for multi-word
                        entity = {
                            "text": phrase,
                            "type": entity_type,
                            "confidence": confidence,
                            "position": i,
                            "length": j,
                            "normalized": self.normalize_arabic_text(phrase)
                        }
                        
                        entities.append(entity)
        
        # Remove overlapping entities (keep highest confidence)
        entities = self._remove_overlapping_entities(entities)
        
        return entities
    
    def _is_potential_entity(self, text: str) -> bool:
        """Check if text could be an entity"""
        text = text.strip()
        
        # Skip very short or very long text
        if len(text) < 2 or len(text) > 50:
            return False
        
        # Skip common stop words
        arabic_stopwords = [
            "في", "من", "إلى", "على", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
            "التي", "الذي", "التي", "كان", "كانت", "يكون", "تكون", "لكن", "لكن"
        ]
        
        if text in arabic_stopwords:
            return False
        
        # Must contain Arabic characters
        has_arabic = any('\u0600' <= char <= '\u06FF' for char in text)
        if not has_arabic:
            return False
        
        return True
    
    def _remove_overlapping_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping entities, keeping highest confidence"""
        if not entities:
            return []
        
        # Sort by confidence (descending)
        entities.sort(key=lambda x: x["confidence"], reverse=True)
        
        filtered_entities = []
        used_positions = set()
        
        for entity in entities:
            start_pos = entity["position"]
            end_pos = start_pos + entity["length"]
            
            # Check for overlap
            overlap = any(pos in used_positions for pos in range(start_pos, end_pos))
            
            if not overlap:
                filtered_entities.append(entity)
                used_positions.update(range(start_pos, end_pos))
        
        # Sort by position
        filtered_entities.sort(key=lambda x: x["position"])
        
        return filtered_entities
    
    def link_entities(self, entities: List[Dict[str, Any]], 
                     context: str = "") -> List[Dict[str, Any]]:
        """
        Link entities to knowledge base and add relationship information
        
        Args:
            entities: List of extracted entities
            context: Additional context for linking
            
        Returns:
            List of linked entities with additional metadata
        """
        linked_entities = []
        
        for entity in entities:
            linked_entity = entity.copy()
            
            # Add knowledge base information
            normalized_text = entity["normalized"]
            if normalized_text in self.entity_kb:
                kb_info = self.entity_kb[normalized_text]
                linked_entity.update(kb_info)
                linked_entity["kb_match"] = True
            else:
                linked_entity["kb_match"] = False
            
            # Add relationship potential
            linked_entity["relationship_potential"] = self._calculate_relationship_potential(
                entity, entities, context
            )
            
            # Add Arabic-specific features
            linked_entity["arabic_features"] = self._extract_arabic_features(entity["text"])
            
            linked_entities.append(linked_entity)
        
        return linked_entities
    
    def _calculate_relationship_potential(self, entity: Dict[str, Any],
                                       all_entities: List[Dict[str, Any]],
                                       context: str) -> Dict[str, float]:
        """Calculate potential relationships with other entities"""
        potential = {}
        
        entity_type = entity["type"]
        
        for other_entity in all_entities:
            if other_entity == entity:
                continue
            
            other_type = other_entity["type"]
            relationship_key = f"{entity_type}-{other_type}"
            
            # Calculate relationship probability based on types
            if entity_type == "person" and other_type == "person":
                potential[other_entity["text"]] = 0.7  # Family/social relations
            elif entity_type == "person" and other_type == "location":
                potential[other_entity["text"]] = 0.8  # Lives in/works at
            elif entity_type == "person" and other_type == "object":
                potential[other_entity["text"]] = 0.6  # Owns/uses
            elif entity_type == "person" and other_type == "organization":
                potential[other_entity["text"]] = 0.7  # Works for/studies at
            else:
                potential[other_entity["text"]] = 0.4  # Generic relationship
        
        return potential
    
    def _extract_arabic_features(self, text: str) -> Dict[str, Any]:
        """Extract Arabic-specific linguistic features"""
        features = {
            "has_definite_article": text.startswith("ال"),
            "has_diacritics": any(char in self.arabic_diacritics for char in text),
            "word_count": len(text.split()),
            "character_count": len(text),
            "is_rtl": True,  # Arabic is always RTL
            "script": "arabic"
        }
        
        # Check for specific Arabic patterns
        features["has_person_prefix"] = any(text.startswith(prefix) for prefix in self.arabic_patterns["person_prefixes"])
        features["has_location_prefix"] = any(text.startswith(prefix) for prefix in self.arabic_patterns["location_prefixes"])
        features["has_org_prefix"] = any(text.startswith(prefix) for prefix in self.arabic_patterns["org_prefixes"])
        
        return features
    
    def predict_relation(self, source_entity: str, target_entity: str,
                        context: str = "") -> Tuple[str, float]:
        """
        Predict the most likely relation between two Arabic entities
        
        Args:
            source_entity: Source entity text
            target_entity: Target entity text
            context: Additional context
            
        Returns:
            Tuple of (predicted_relation, confidence_score)
        """
        source_type, _ = self.detect_entity_type(source_entity)
        target_type, _ = self.detect_entity_type(target_entity)
        
        # Normalize entities
        source_norm = self.normalize_arabic_text(source_entity)
        target_norm = self.normalize_arabic_text(target_entity)
        
        # Check for explicit relations in context
        context_norm = self.normalize_arabic_text(context)
        
        for relation, info in self.arabic_relations.items():
            if relation in context_norm:
                return relation, info["strength"]
        
        # Predict based on entity types
        type_pair = f"{source_type}-{target_type}"
        
        relation_predictions = {
            "person-person": ("صديق", 0.6),
            "person-location": ("يعيش في", 0.7),
            "person-object": ("يملك", 0.6),
            "person-organization": ("يعمل في", 0.7),
            "location-location": ("بجانب", 0.5),
            "organization-location": ("يقع في", 0.8)
        }
        
        if type_pair in relation_predictions:
            return relation_predictions[type_pair]
        
        # Default prediction
        return "مرتبط بـ", 0.4
    
    def get_entity_info(self, entity: str) -> Dict[str, Any]:
        """
        Get comprehensive information about an Arabic entity
        
        Args:
            entity: Entity text
            
        Returns:
            Dictionary with entity information
        """
        normalized = self.normalize_arabic_text(entity)
        entity_type, confidence = self.detect_entity_type(entity)
        
        info = {
            "original_text": entity,
            "normalized_text": normalized,
            "type": entity_type,
            "type_confidence": confidence,
            "arabic_features": self._extract_arabic_features(entity),
            "kb_match": normalized in self.entity_kb
        }
        
        # Add knowledge base information if available
        if info["kb_match"]:
            info.update(self.entity_kb[normalized])
        
        # Add potential relations
        info["potential_relations"] = list(self.arabic_relations.keys())
        
        return info
    
    def create_arabic_graph_edges(self, entities: List[Dict[str, Any]],
                                 context: str = "") -> List[Dict[str, Any]]:
        """
        Create graph edges between Arabic entities
        
        Args:
            entities: List of linked entities
            context: Context for relation prediction
            
        Returns:
            List of graph edges with Arabic relations
        """
        edges = []
        
        for i, source_entity in enumerate(entities):
            for j, target_entity in enumerate(entities):
                if i >= j:  # Avoid duplicates and self-loops
                    continue
                
                source_text = source_entity["text"]
                target_text = target_entity["text"]
                
                # Predict relation
                relation, confidence = self.predict_relation(source_text, target_text, context)
                
                # Only create edge if confidence is above threshold
                if confidence > 0.5:
                    edge = {
                        "source": source_text,
                        "target": target_text,
                        "relation": relation,
                        "confidence": confidence,
                        "source_type": source_entity["type"],
                        "target_type": target_entity["type"],
                        "bidirectional": self.arabic_relations.get(relation, {}).get("bidirectional", False),
                        "relation_type": self.arabic_relations.get(relation, {}).get("type", "unknown")
                    }
                    
                    edges.append(edge)
        
        return edges
    
    def process_arabic_query(self, query: str) -> Dict[str, Any]:
        """
        Process Arabic query for entity linking and relation extraction
        
        Args:
            query: Arabic query text
            
        Returns:
            Processed query information
        """
        normalized_query = self.normalize_arabic_text(query)
        
        # Extract entities from query
        entities = self.extract_entities(normalized_query)
        
        # Link entities
        linked_entities = self.link_entities(entities, normalized_query)
        
        # Create potential graph edges
        edges = self.create_arabic_graph_edges(linked_entities, normalized_query)
        
        # Identify query intent
        query_intent = self._identify_query_intent(normalized_query)
        
        return {
            "original_query": query,
            "normalized_query": normalized_query,
            "entities": linked_entities,
            "potential_edges": edges,
            "query_intent": query_intent,
            "language": "ar",
            "rtl": True
        }
    
    def _identify_query_intent(self, query: str) -> Dict[str, Any]:
        """Identify the intent of Arabic query"""
        intent_patterns = {
            "who_owns": ["من يملك", "من لديه", "صاحب"],
            "what_is": ["ما هو", "ماذا", "أي"],
            "where_is": ["أين", "في أي مكان", "مكان"],
            "how_does": ["كيف", "بأي طريقة"],
            "when_did": ["متى", "في أي وقت"],
            "why_does": ["لماذا", "ما السبب"]
        }
        
        detected_intents = []
        
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    detected_intents.append(intent)
                    break
        
        return {
            "primary_intent": detected_intents[0] if detected_intents else "unknown",
            "all_intents": detected_intents,
            "confidence": 0.8 if detected_intents else 0.3
        }