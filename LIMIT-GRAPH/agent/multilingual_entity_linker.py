# -*- coding: utf-8 -*-
"""
LIMIT-GRAPH v1.1: Multilingual Entity Linker
Enhanced entity linking with cross-lingual support and semantic graph preservation
"""

import json
import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from fuzzywuzzy import process, fuzz
import networkx as nx
import logging

@dataclass
class EntityMatch:
    """Represents an entity match with confidence and metadata"""
    entity: str
    graph_node: str
    confidence: float
    lang: str
    wikidata_qid: Optional[str] = None
    semantic_type: Optional[str] = None

@dataclass
class GraphTraversalPath:
    """Represents a reasoning path through the semantic graph"""
    start_node: str
    end_node: str
    path: List[str]
    relations: List[str]
    confidence: float
    lang: str

class MultilingualEntityLinker:
    """
    Enhanced multilingual entity linker for LIMIT-GRAPH v1.1
    Supports cross-lingual evaluation and semantic graph preservation
    """
    
    def __init__(self, graph_edges: List[Dict], lang: str = 'id'):
        self.lang = lang
        self.graph_edges = graph_edges
        self.graph = self._build_graph()
        self.entity_mappings = self._load_entity_mappings()
        self.logger = logging.getLogger(__name__)
        
        # Language-specific patterns for entity extraction
        self.lang_patterns = {
            'id': {
                'person_indicators': ['yang', 'siapa', 'orang'],
                'relation_patterns': ['menyukai', 'alergi terhadap', 'suka'],
                'question_words': ['siapa', 'apa', 'dimana', 'kapan', 'mengapa']
            },
            'es': {
                'person_indicators': ['quien', 'persona', 'gente'],
                'relation_patterns': ['le gusta', 'alérgico a', 'ama'],
                'question_words': ['quien', 'que', 'donde', 'cuando', 'por que']
            },
            'en': {
                'person_indicators': ['who', 'person', 'people'],
                'relation_patterns': ['likes', 'allergic to', 'loves'],
                'question_words': ['who', 'what', 'where', 'when', 'why']
            }
        }
    
    def _build_graph(self) -> nx.DiGraph:
        """Build NetworkX graph from edges"""
        G = nx.DiGraph()
        for edge in self.graph_edges:
            if edge.get('lang') == self.lang:
                G.add_edge(
                    edge['source'], 
                    edge['target'], 
                    relation=edge['relation'],
                    lang=edge['lang']
                )
        return G
    
    def _load_entity_mappings(self) -> Dict[str, Dict]:
        """Load cross-lingual entity mappings (Wikidata QIDs, etc.)"""
        # In a real implementation, this would load from a mapping file
        return {
            'Andrew': {'id': 'Andrew', 'es': 'Andrés', 'en': 'Andrew', 'qid': 'Q123'},
            'Joanna': {'id': 'Joanna', 'es': 'Juana', 'en': 'Joanna', 'qid': 'Q456'},
            'apel': {'id': 'apel', 'es': 'manzana', 'en': 'apple', 'qid': 'Q89'}
        }
    
    def extract_entities(self, query: str) -> List[str]:
        """Extract entities from query using language-specific patterns"""
        entities = []
        
        # Simple entity extraction based on graph nodes
        graph_nodes = list(self.graph.nodes())
        
        # Find exact matches first
        for node in graph_nodes:
            if node.lower() in query.lower():
                entities.append(node)
        
        # Find partial matches using fuzzy matching
        words = query.split()
        for word in words:
            if len(word) > 2:  # Skip short words
                matches = process.extract(word, graph_nodes, limit=3, scorer=fuzz.ratio)
                for match, score in matches:
                    if score > 70 and match not in entities:
                        entities.append(match)
        
        return list(set(entities))  # Remove duplicates
    
    def link_entities(self, entities: List[str]) -> List[EntityMatch]:
        """Link extracted entities to graph nodes with confidence scores"""
        linked_entities = []
        graph_nodes = list(self.graph.nodes())
        
        for entity in entities:
            # Direct match
            if entity in graph_nodes:
                match = EntityMatch(
                    entity=entity,
                    graph_node=entity,
                    confidence=1.0,
                    lang=self.lang,
                    wikidata_qid=self._get_wikidata_qid(entity)
                )
                linked_entities.append(match)
            else:
                # Fuzzy match
                best_match, score = process.extractOne(entity, graph_nodes)
                if score > 60:  # Lower threshold for multilingual
                    match = EntityMatch(
                        entity=entity,
                        graph_node=best_match,
                        confidence=score / 100.0,
                        lang=self.lang,
                        wikidata_qid=self._get_wikidata_qid(best_match)
                    )
                    linked_entities.append(match)
        
        return linked_entities
    
    def _get_wikidata_qid(self, entity: str) -> Optional[str]:
        """Get Wikidata QID for entity if available"""
        for canonical, mappings in self.entity_mappings.items():
            if entity in mappings.values():
                return mappings.get('qid')
        return None
    
    def traverse_graph(self, start_entities: List[str], max_hops: int = 3) -> List[GraphTraversalPath]:
        """Traverse semantic graph to find reasoning paths"""
        paths = []
        
        for start_entity in start_entities:
            if start_entity not in self.graph:
                continue
                
            # Find all paths within max_hops
            for target in self.graph.nodes():
                if start_entity == target:
                    continue
                    
                try:
                    # Find shortest path
                    path = nx.shortest_path(self.graph, start_entity, target)
                    if len(path) <= max_hops + 1:
                        # Extract relations along the path
                        relations = []
                        for i in range(len(path) - 1):
                            edge_data = self.graph.get_edge_data(path[i], path[i + 1])
                            if edge_data:
                                relations.append(edge_data.get('relation', 'unknown'))
                        
                        traversal_path = GraphTraversalPath(
                            start_node=start_entity,
                            end_node=target,
                            path=path,
                            relations=relations,
                            confidence=1.0 / len(path),  # Inverse of path length
                            lang=self.lang
                        )
                        paths.append(traversal_path)
                        
                except nx.NetworkXNoPath:
                    continue
        
        return sorted(paths, key=lambda x: x.confidence, reverse=True)
    
    def get_multilingual_alignment(self, entity: str, target_lang: str) -> Optional[str]:
        """Get entity alignment for target language"""
        for canonical, mappings in self.entity_mappings.items():
            if entity in mappings.values():
                return mappings.get(target_lang)
        return None
    
    def validate_graph_structure(self) -> Dict[str, any]:
        """Validate semantic graph structure preservation"""
        validation_results = {
            'node_count': len(self.graph.nodes()),
            'edge_count': len(self.graph.edges()),
            'connected_components': nx.number_weakly_connected_components(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / len(self.graph.nodes()) if self.graph.nodes() else 0,
            'lang': self.lang,
            'has_cycles': not nx.is_directed_acyclic_graph(self.graph)
        }
        
        return validation_results
    
    def process_query(self, query: str) -> Dict[str, any]:
        """Complete query processing pipeline"""
        # Extract entities
        entities = self.extract_entities(query)
        
        # Link entities
        linked_entities = self.link_entities(entities)
        
        # Traverse graph
        entity_nodes = [match.graph_node for match in linked_entities]
        traversal_paths = self.traverse_graph(entity_nodes)
        
        return {
            'query': query,
            'lang': self.lang,
            'extracted_entities': entities,
            'linked_entities': [
                {
                    'entity': match.entity,
                    'graph_node': match.graph_node,
                    'confidence': match.confidence,
                    'wikidata_qid': match.wikidata_qid
                } for match in linked_entities
            ],
            'traversal_paths': [
                {
                    'start_node': path.start_node,
                    'end_node': path.end_node,
                    'path': path.path,
                    'relations': path.relations,
                    'confidence': path.confidence
                } for path in traversal_paths[:5]  # Top 5 paths
            ],
            'graph_stats': self.validate_graph_structure()
        }
