"""
Axiom Hive Core Logic - Simplified Implementation
Implements VCS (Vector Constrained Singularity) and Hamiltonian Validator
"""

import hashlib
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json
from datetime import datetime


class QuantizationResult(Enum):
    """Possible quantization results"""
    ACCEPTED = "ACCEPTED"
    REJECTED_OUT_OF_BOUNDS = "REJECTED_OUT_OF_BOUNDS"
    REJECTED_AMBIGUOUS = "REJECTED_AMBIGUOUS"
    ERROR_INVALID_INPUT = "ERROR_INVALID_INPUT"


@dataclass
class StateVector:
    """Represents a discrete, valid state within the system"""
    coordinates: np.ndarray
    label: str
    metadata: Optional[Dict] = None
    
    def distance_to(self, point: np.ndarray) -> float:
        """Compute Euclidean distance to point"""
        return float(np.linalg.norm(self.coordinates - point))


@dataclass
class QuantizationOutput:
    """Complete output from VCS quantization"""
    result: QuantizationResult
    state_label: Optional[str]
    residual_energy: float
    nearest_states: List[Tuple[str, float]]
    input_hash: str


@dataclass
class LogicNode:
    """Represents a proposition in reasoning chain"""
    proposition: str
    truth_value: float
    evidence_ids: List[str]
    node_id: str


@dataclass
class ValidationResult:
    """Result of Hamiltonian validation"""
    is_valid: bool
    lambda_score: float
    energy_level: float
    violations: List[str]
    trace_hash: str


class VectorConstrainedSingularity:
    """
    The Fold: Maps continuous inputs to discrete valid states.
    Production-ready implementation with comprehensive error handling.
    """
    
    def __init__(
        self,
        threshold: float = 0.05,
        max_states: int = 10000,
        tie_break_strategy: str = "lexicographic"
    ):
        """
        Initialize VCS engine.
        
        Args:
            threshold: Maximum allowed distance for state acceptance
            max_states: Maximum number of valid states
            tie_break_strategy: How to handle equidistant states
        """
        self.threshold = threshold
        self.max_states = max_states
        self.tie_break_strategy = tie_break_strategy
        self.valid_states: List[StateVector] = []
    
    def define_invariant(
        self, 
        coordinates: List[float], 
        label: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """Add a valid state to the lattice."""
        if len(self.valid_states) >= self.max_states:
            raise ValueError(f"Cannot exceed {self.max_states} states")
            
        if any(s.label == label for s in self.valid_states):
            raise ValueError(f"Duplicate state label: {label}")
            
        state = StateVector(
            coordinates=np.array(coordinates, dtype=np.float64),
            label=label,
            metadata=metadata or {}
        )
        self.valid_states.append(state)
    
    def fold(self, input_vector: List[float]) -> QuantizationOutput:
        """Quantize input to nearest valid state."""
        try:
            input_arr = np.array(input_vector, dtype=np.float64)
        except (ValueError, TypeError) as e:
            return QuantizationOutput(
                result=QuantizationResult.ERROR_INVALID_INPUT,
                state_label=None,
                residual_energy=float('inf'),
                nearest_states=[],
                input_hash=self._hash_input(input_vector)
            )
        
        if len(self.valid_states) == 0:
            return QuantizationOutput(
                result=QuantizationResult.REJECTED_OUT_OF_BOUNDS,
                state_label=None,
                residual_energy=float('inf'),
                nearest_states=[],
                input_hash=self._hash_input(input_arr)
            )
        
        # Compute distances to all states
        distances = [
            (state.label, state.distance_to(input_arr))
            for state in self.valid_states
        ]
        distances.sort(key=lambda x: x[1])
        
        min_distance = distances[0][1]
        nearest_label = distances[0][0]
        
        # Check for within threshold
        if min_distance > self.threshold:
            return QuantizationOutput(
                result=QuantizationResult.REJECTED_OUT_OF_BOUNDS,
                state_label=None,
                residual_energy=min_distance,
                nearest_states=distances[:3],
                input_hash=self._hash_input(input_arr)
            )
        
        # Check for ties
        ties = [d for d in distances if d[1] <= self.threshold]
        
        if len(ties) > 1:
            if self.tie_break_strategy == "reject":
                return QuantizationOutput(
                    result=QuantizationResult.REJECTED_AMBIGUOUS,
                    state_label=None,
                    residual_energy=min_distance,
                    nearest_states=ties,
                    input_hash=self._hash_input(input_arr)
                )
            elif self.tie_break_strategy == "lexicographic":
                nearest_label = sorted([t[0] for t in ties])[0]
            elif self.tie_break_strategy == "first":
                nearest_label = ties[0][0]
        
        # Successful quantization
        return QuantizationOutput(
            result=QuantizationResult.ACCEPTED,
            state_label=nearest_label,
            residual_energy=0.0,
            nearest_states=distances[:5],
            input_hash=self._hash_input(input_arr)
        )
    
    def _hash_input(self, input_arr) -> str:
        """Generate SHA-256 hash of input for receipts"""
        input_bytes = np.array(input_arr).tobytes()
        return hashlib.sha256(input_bytes).hexdigest()[:16]


class HamiltonianValidator:
    """
    Enforces energy conservation and logical consistency.
    """
    
    def __init__(
        self,
        energy_ceiling: float = 100.0,
        lambda_threshold: float = 1e-9
    ):
        """Initialize validator."""
        self.energy_ceiling = energy_ceiling
        self.lambda_threshold = lambda_threshold
        self.current_energy = 0.0
    
    def validate_energy(self, proposed_energy: float) -> bool:
        """Check if proposed energy violates constraints."""
        if proposed_energy > self.energy_ceiling:
            return False
        
        if proposed_energy > self.current_energy:
            return False
        
        self.current_energy = proposed_energy
        return True
    
    def validate_logic_chain(
        self, 
        nodes: List[LogicNode]
    ) -> ValidationResult:
        """Validate logical consistency of reasoning chain."""
        violations = []
        
        # Build dependency graph
        node_map = {n.node_id: n for n in nodes}
        
        # Check for cycles
        if self._has_cycle(nodes, node_map):
            violations.append("CYCLE_DETECTED: Circular reasoning found")
        
        # Validate each node's truth value against evidence
        lambda_scores = []
        
        for node in nodes:
            if not node.evidence_ids:
                lambda_scores.append(1.0)
                continue
            
            # Get evidence nodes
            evidence_nodes = [
                node_map[eid] for eid in node.evidence_ids 
                if eid in node_map
            ]
            
            if len(evidence_nodes) != len(node.evidence_ids):
                violations.append(f"MISSING_EVIDENCE: Node {node.node_id}")
                lambda_scores.append(0.0)
                continue
            
            # Compute Λ
            evidence_strength = sum(e.truth_value for e in evidence_nodes)
            claim_strength = node.truth_value
            
            if claim_strength == 0:
                lambda_scores.append(1.0)
                continue
            
            lambda_score = evidence_strength / claim_strength
            lambda_scores.append(lambda_score)
            
            # Check if Λ ≈ 1.0
            if abs(lambda_score - 1.0) > self.lambda_threshold:
                violations.append(
                    f"INCONSISTENT: Node {node.node_id} has Λ={lambda_score:.6f}"
                )
        
        # Overall Λ score
        overall_lambda = sum(lambda_scores) / len(lambda_scores) if lambda_scores else 0.0
        
        # Compute energy
        energy = float(len(nodes))
        
        is_valid = len(violations) == 0 and overall_lambda >= (1.0 - self.lambda_threshold)
        
        return ValidationResult(
            is_valid=is_valid,
            lambda_score=overall_lambda,
            energy_level=energy,
            violations=violations,
            trace_hash=self._hash_chain(nodes)
        )
    
    def _has_cycle(self, nodes: List[LogicNode], node_map: Dict) -> bool:
        """Detect cycles in dependency graph (DFS)"""
        visited = set()
        rec_stack = set()
        
        def visit(node_id):
            if node_id in rec_stack:
                return True
            if node_id in visited:
                return False
            
            visited.add(node_id)
            rec_stack.add(node_id)
            
            node = node_map.get(node_id)
            if node:
                for dep_id in node.evidence_ids:
                    if visit(dep_id):
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        for node in nodes:
            if visit(node.node_id):
                return True
        
        return False
    
    def _hash_chain(self, nodes: List[LogicNode]) -> str:
        """Generate hash of logic chain for receipts"""
        chain_str = "|".join([
            f"{n.node_id}:{n.proposition}:{n.truth_value}" 
            for n in nodes
        ])
        return hashlib.sha256(chain_str.encode()).hexdigest()[:16]


class AxiomHive:
    """Main Axiom Hive system combining VCS and Hamiltonian"""
    
    def __init__(self, vcs: VectorConstrainedSingularity, hamiltonian: HamiltonianValidator):
        """Initialize Axiom Hive with VCS and Hamiltonian"""
        self.vcs = vcs
        self.hamiltonian = hamiltonian
    
    def process(self, input_vector: List[float]) -> Dict:
        """Process input and return result with receipt"""
        # Quantize input
        quant_result = self.vcs.fold(input_vector)
        
        # Generate receipt
        receipt = {
            "timestamp": datetime.now().isoformat(),
            "input_hash": quant_result.input_hash,
            "result": quant_result.result.value,
            "state_label": quant_result.state_label,
            "residual_energy": quant_result.residual_energy,
            "nearest_states": quant_result.nearest_states[:3]
        }
        
        return receipt
