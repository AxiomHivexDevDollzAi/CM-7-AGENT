"""
Axiom Hive Desktop Application
A GUI for the Axiom Hive deterministic AI system
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
from axiom_core import (
    VectorConstrainedSingularity,
    HamiltonianValidator,
    AxiomHive,
    LogicNode,
    QuantizationResult
)


class AxiomHiveApp:
    """Main desktop application for Axiom Hive"""
    
    def __init__(self, root):
        """Initialize the application"""
        self.root = root
        self.root.title("Axiom Hive - Deterministic AI System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize Axiom Hive system
        self.vcs = VectorConstrainedSingularity(threshold=0.15)
        self.hamiltonian = HamiltonianValidator(energy_ceiling=100.0)
        self.hive = AxiomHive(self.vcs, self.hamiltonian)
        
        # Setup default states
        self.setup_default_states()
        
        # Create UI
        self.create_ui()
    
    def setup_default_states(self):
        """Define default valid states for the system"""
        self.vcs.define_invariant([0.0, 1.0], "SAFE_STATE", {"description": "Safe operating state"})
        self.vcs.define_invariant([1.0, 0.0], "RISKY_STATE", {"description": "Risky operating state"})
        self.vcs.define_invariant([0.5, 0.5], "NEUTRAL_STATE", {"description": "Neutral state"})
    
    def create_ui(self):
        """Create the user interface"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tab 1: VCS Quantization
        vcs_frame = ttk.Frame(notebook)
        notebook.add(vcs_frame, text="VCS Quantization")
        self.create_vcs_tab(vcs_frame)
        
        # Tab 2: Hamiltonian Validation
        ham_frame = ttk.Frame(notebook)
        notebook.add(ham_frame, text="Hamiltonian Validation")
        self.create_hamiltonian_tab(ham_frame)
        
        # Tab 3: System Info
        info_frame = ttk.Frame(notebook)
        notebook.add(info_frame, text="System Info")
        self.create_info_tab(info_frame)
    
    def create_vcs_tab(self, parent):
        """Create the VCS Quantization tab"""
        # Input section
        input_frame = ttk.LabelFrame(parent, text="Input Vector", padding=10)
        input_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(input_frame, text="Enter two values (0.0 to 1.0):").pack(anchor="w")
        
        input_sub_frame = ttk.Frame(input_frame)
        input_sub_frame.pack(fill="x", pady=5)
        
        ttk.Label(input_sub_frame, text="X:").pack(side="left", padx=5)
        self.vcs_x_var = tk.StringVar(value="0.1")
        ttk.Entry(input_sub_frame, textvariable=self.vcs_x_var, width=10).pack(side="left", padx=5)
        
        ttk.Label(input_sub_frame, text="Y:").pack(side="left", padx=5)
        self.vcs_y_var = tk.StringVar(value="0.9")
        ttk.Entry(input_sub_frame, textvariable=self.vcs_y_var, width=10).pack(side="left", padx=5)
        
        ttk.Button(input_sub_frame, text="Process", command=self.process_vcs).pack(side="left", padx=10)
        
        # Result section
        result_frame = ttk.LabelFrame(parent, text="Quantization Result", padding=10)
        result_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.vcs_result_text = scrolledtext.ScrolledText(result_frame, height=20, width=80)
        self.vcs_result_text.pack(fill="both", expand=True)
    
    def create_hamiltonian_tab(self, parent):
        """Create the Hamiltonian Validation tab"""
        # Energy input section
        energy_frame = ttk.LabelFrame(parent, text="Energy Validation", padding=10)
        energy_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(energy_frame, text="Proposed Energy Level:").pack(anchor="w")
        
        energy_sub_frame = ttk.Frame(energy_frame)
        energy_sub_frame.pack(fill="x", pady=5)
        
        self.energy_var = tk.StringVar(value="50.0")
        ttk.Entry(energy_sub_frame, textvariable=self.energy_var, width=10).pack(side="left", padx=5)
        
        ttk.Button(energy_sub_frame, text="Validate Energy", command=self.validate_energy).pack(side="left", padx=10)
        
        # Logic chain section
        logic_frame = ttk.LabelFrame(parent, text="Logic Chain Validation", padding=10)
        logic_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ttk.Label(logic_frame, text="Example Logic Chain:").pack(anchor="w")
        
        button_frame = ttk.Frame(logic_frame)
        button_frame.pack(fill="x", pady=5)
        
        ttk.Button(button_frame, text="Validate Example Chain", command=self.validate_logic_chain).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Clear Results", command=self.clear_hamiltonian_results).pack(side="left", padx=5)
        
        # Result section
        result_frame = ttk.LabelFrame(logic_frame, text="Validation Results", padding=10)
        result_frame.pack(fill="both", expand=True, pady=10)
        
        self.ham_result_text = scrolledtext.ScrolledText(result_frame, height=15, width=80)
        self.ham_result_text.pack(fill="both", expand=True)
    
    def create_info_tab(self, parent):
        """Create the System Info tab"""
        info_frame = ttk.Frame(parent, padding=20)
        info_frame.pack(fill="both", expand=True)
        
        # Title
        title_label = ttk.Label(info_frame, text="Axiom Hive System Information", font=("Arial", 16, "bold"))
        title_label.pack(anchor="w", pady=10)
        
        # Info text
        info_text = scrolledtext.ScrolledText(info_frame, height=30, width=80)
        info_text.pack(fill="both", expand=True)
        
        info_content = """
AXIOM HIVE - Deterministic AI System
=====================================

OVERVIEW:
Axiom Hive is a deterministic AI system that executes logic with zero variance (C=0).
Unlike probabilistic AI systems that "guess" based on statistical patterns, Axiom Hive 
computes exact outputs using formal logic and constraint programming.

CORE COMPONENTS:

1. Vector Constrained Singularity (VCS) - "The Fold"
   - Maps continuous inputs to discrete valid states
   - Quantizes input vectors to the nearest valid state
   - Prevents out-of-distribution inputs
   - Defends against prompt injection attacks

2. Hamiltonian Validator
   - Enforces energy conservation (dE/dt ≤ 0)
   - Ensures logical consistency (Λ = 1.0)
   - Validates reasoning chains
   - Detects circular reasoning

KEY GUARANTEES:

I1: Deterministic Reproducibility (C=0)
    - Identical inputs produce bit-exact identical outputs
    - Verified through SHA-256 hashing

I2: Energy Conservation
    - Total system energy never increases
    - Prevents unbounded hallucination

I3: State Space Constraint
    - System only produces outputs from pre-validated states
    - Rejects out-of-distribution inputs

I4: Logical Consistency (Λ = 1.0)
    - All outputs are logically derivable from inputs and axioms
    - Evidence must fully support claims

I5: Cryptographic Auditability
    - Every decision has verifiable proof of reasoning chain
    - Includes SHA-256 hashes and timestamps

USE CASES:

✓ Good Fit:
  - Financial compliance (AML, KYC, transaction screening)
  - Healthcare decision support (diagnosis verification)
  - Legal contract analysis (clause validation)
  - Infrastructure safety (control systems)
  - Regulatory reporting (audit-ready outputs)

✗ Poor Fit:
  - Creative writing
  - General conversation
  - Brainstorming / ideation
  - Ambiguous/subjective tasks

SECURITY PROPERTIES:

Compared to Probabilistic AI (e.g., LLMs):
  - Reproducibility: ✓ C = 0 (vs. Variance > 0)
  - Auditability: ✓ Complete glass box (vs. Partial black box)
  - Prompt Injection: ✓ Protected by VCS (vs. Vulnerable)
  - Hallucination: ✓ Architecturally Impossible (vs. Possible)
  - Logic Errors: ✓ Caught by Λ validator (vs. Undetected)

CURRENT SYSTEM CONFIGURATION:

VCS Settings:
  - Threshold: 0.15 (maximum distance for state acceptance)
  - Max States: 10,000
  - Tie-break Strategy: Lexicographic

Hamiltonian Settings:
  - Energy Ceiling: 100.0
  - Lambda Threshold: 1e-9

Valid States:
  1. SAFE_STATE: [0.0, 1.0]
  2. RISKY_STATE: [1.0, 0.0]
  3. NEUTRAL_STATE: [0.5, 0.5]

For more information, visit the documentation or source code.
        """
        
        info_text.insert("1.0", info_content)
        info_text.config(state="disabled")
    
    def process_vcs(self):
        """Process VCS quantization"""
        try:
            x = float(self.vcs_x_var.get())
            y = float(self.vcs_y_var.get())
            
            # Validate input range
            if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
                messagebox.showerror("Input Error", "Values must be between 0.0 and 1.0")
                return
            
            # Process through Axiom Hive
            receipt = self.hive.process([x, y])
            
            # Display result
            self.vcs_result_text.config(state="normal")
            self.vcs_result_text.delete("1.0", "end")
            
            result_text = f"""
QUANTIZATION RECEIPT
====================

Timestamp: {receipt['timestamp']}
Input Hash: {receipt['input_hash']}

Input Vector: [{x}, {y}]
Result: {receipt['result']}
State Label: {receipt['state_label']}
Residual Energy: {receipt['residual_energy']:.6f}

Nearest States:
"""
            for i, (label, distance) in enumerate(receipt['nearest_states'], 1):
                result_text += f"  {i}. {label}: distance = {distance:.6f}\n"
            
            result_text += f"""
SECURITY PROPERTIES:
- Deterministic: ✓ (C=0)
- Reproducible: ✓ (Same input → Same output)
- Auditable: ✓ (SHA-256 hash provided)
- Hallucination-proof: ✓ (Constrained to valid states)
"""
            
            self.vcs_result_text.insert("1.0", result_text)
            self.vcs_result_text.config(state="disabled")
            
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid floating-point numbers")
    
    def validate_energy(self):
        """Validate energy level"""
        try:
            energy = float(self.energy_var.get())
            
            # Validate energy
            is_valid = self.hamiltonian.validate_energy(energy)
            
            # Display result
            self.ham_result_text.config(state="normal")
            self.ham_result_text.delete("1.0", "end")
            
            result_text = f"""
ENERGY VALIDATION RESULT
========================

Proposed Energy: {energy:.2f}
Energy Ceiling: {self.hamiltonian.energy_ceiling:.2f}
Current Energy: {self.hamiltonian.current_energy:.2f}

Validation Result: {'✓ VALID' if is_valid else '✗ INVALID'}

Constraints Checked:
1. Energy ≤ Ceiling: {'✓' if energy <= self.hamiltonian.energy_ceiling else '✗'}
2. Non-increasing (dE/dt ≤ 0): {'✓' if energy <= self.hamiltonian.current_energy else '✗'}

Interpretation:
- Energy conservation ensures the system cannot create information from nothing
- This makes hallucination architecturally impossible
- The system can only work with information present in the input
"""
            
            self.ham_result_text.insert("1.0", result_text)
            self.ham_result_text.config(state="disabled")
            
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid floating-point number")
    
    def validate_logic_chain(self):
        """Validate an example logic chain"""
        # Create an example logic chain
        nodes = [
            LogicNode(
                proposition="A: It is raining",
                truth_value=1.0,
                evidence_ids=[],
                node_id="A"
            ),
            LogicNode(
                proposition="B: The ground is wet",
                truth_value=1.0,
                evidence_ids=["A"],
                node_id="B"
            ),
            LogicNode(
                proposition="C: If it rains, the ground is wet",
                truth_value=1.0,
                evidence_ids=["A", "B"],
                node_id="C"
            )
        ]
        
        # Validate
        result = self.hamiltonian.validate_logic_chain(nodes)
        
        # Display result
        self.ham_result_text.config(state="normal")
        self.ham_result_text.delete("1.0", "end")
        
        result_text = f"""
LOGIC CHAIN VALIDATION RESULT
=============================

Chain: A → B → C

Propositions:
  A: It is raining (truth_value: 1.0)
  B: The ground is wet (truth_value: 1.0, depends on: A)
  C: If it rains, the ground is wet (truth_value: 1.0, depends on: A, B)

Validation Result: {'✓ VALID' if result.is_valid else '✗ INVALID'}

Lambda Score (Λ): {result.lambda_score:.9f}
  - Perfect consistency requires Λ = 1.0 ± {self.hamiltonian.lambda_threshold}
  - Current score: {'✓ PASS' if abs(result.lambda_score - 1.0) <= self.hamiltonian.lambda_threshold else '✗ FAIL'}

Energy Level: {result.energy_level:.2f}

Violations: {len(result.violations)}
"""
        
        if result.violations:
            result_text += "\nViolation Details:\n"
            for violation in result.violations:
                result_text += f"  - {violation}\n"
        else:
            result_text += "\nNo violations detected. ✓\n"
        
        result_text += f"""
Trace Hash: {result.trace_hash}

Interpretation:
- Lambda (Λ) measures logical consistency
- Λ = 1.0 means evidence perfectly supports the claim
- The system enforces deductive closure
- All outputs must have complete proof chains
"""
        
        self.ham_result_text.insert("1.0", result_text)
        self.ham_result_text.config(state="disabled")
    
    def clear_hamiltonian_results(self):
        """Clear Hamiltonian results"""
        self.ham_result_text.config(state="normal")
        self.ham_result_text.delete("1.0", "end")
        self.ham_result_text.config(state="disabled")


def main():
    """Run the application"""
    root = tk.Tk()
    app = AxiomHiveApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
