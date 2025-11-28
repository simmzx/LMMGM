"""
Chemical Knowledge Analyzer - Track 2 of Dual-Track Fragment Analysis System

This module uses LLM to analyze fragment chemical properties based on
medicinal chemistry knowledge, independent of NCI detection results.

Key analysis dimensions:
1. Pharmacophore quality (药效团质量)
2. Electronic properties (电子性质)
3. Pocket complementarity (口袋互补性)
4. Drug-likeness (类药性)
5. Modifiability/SAR potential (可修饰性)
6. Binding anchor potential (结合锚点潜力)

This provides a complementary view to structure-based NCI analysis,
enabling more robust fragment selection for diffusion model conditioning.

Author: Enhanced for diffusion model conditioning
"""

import os
import json
import requests
from typing import List, Dict, Optional, Tuple


class ChemicalKnowledgeAnalyzer:
    """
    Track 2: LLM-based chemical knowledge analysis.
    
    This analyzer evaluates fragment importance based on medicinal chemistry
    principles rather than just detected interactions. It considers:
    
    1. Intrinsic chemical properties of the fragment
    2. Known pharmacophore patterns
    3. Pocket environment compatibility
    4. Drug design heuristics
    
    This is complementary to Track 1 (soft NCI) and helps identify fragments
    that may be important due to chemical knowledge not captured in 3D geometry.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the analyzer.
        
        Args:
            api_key: DeepSeek API key (if None, reads from environment)
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("API key required. Set DEEPSEEK_API_KEY environment variable.")
        
        self.api_base = "https://api.deepseek.com/v1"
        self.model = "deepseek-chat"
    
    def _call_llm(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.2,
        max_tokens: int = 3000
    ) -> Optional[str]:
        """Call DeepSeek API."""
        url = f"{self.api_base}/chat/completions"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=90)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"❌ LLM API call failed: {e}")
            return None
    
    def build_knowledge_prompt(
        self,
        fragments: List[Dict],
        pocket_features: Dict,
        nci_summary: Optional[Dict] = None
    ) -> str:
        """
        Build prompt for chemical knowledge analysis.
        
        The prompt is designed to:
        1. NOT rely primarily on NCI counts
        2. Focus on intrinsic chemical properties
        3. Apply medicinal chemistry knowledge
        4. Consider pocket environment
        
        Args:
            fragments: List of fragment data [{smiles, nci_counts, ...}]
            pocket_features: Pocket characteristics
            nci_summary: Optional NCI summary (for reference only)
            
        Returns:
            Formatted prompt string
        """
        
        # Format pocket information
        residues = pocket_features.get('residues', [])
        if isinstance(residues, list):
            residue_str = ', '.join(residues[:20])  # Limit for prompt length
            if len(residues) > 20:
                residue_str += f"... ({len(residues)} total)"
        else:
            residue_str = str(residues)
        
        # Categorize residues
        polar_residues = []
        hydrophobic_residues = []
        charged_residues = []
        aromatic_residues = []
        
        for res in residues if isinstance(residues, list) else []:
            res_name = res[:3] if len(res) >= 3 else res
            if res_name in ['SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS']:
                polar_residues.append(res)
            elif res_name in ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PRO']:
                hydrophobic_residues.append(res)
            elif res_name in ['LYS', 'ARG', 'HIS', 'ASP', 'GLU']:
                charged_residues.append(res)
            if res_name in ['PHE', 'TYR', 'TRP', 'HIS']:
                aromatic_residues.append(res)
        
        prompt = f"""You are an expert medicinal chemist with deep knowledge of drug design, structure-activity relationships (SAR), and protein-ligand interactions.

## IMPORTANT INSTRUCTIONS

Your task is to analyze each fragment's **INTRINSIC CHEMICAL PROPERTIES** and **POTENTIAL for binding**, based on MEDICINAL CHEMISTRY KNOWLEDGE.

**DO NOT** simply rank fragments by NCI counts. The NCI data is provided for reference only.
**DO** evaluate fragments based on their chemical nature, pharmacophore features, and compatibility with the binding pocket.

## Binding Pocket Context

**Pocket Residues**: {residue_str}
**Pocket Characteristics**:
- Polar residues ({len(polar_residues)}): {', '.join(polar_residues[:10]) if polar_residues else 'None detected'}
- Hydrophobic residues ({len(hydrophobic_residues)}): {', '.join(hydrophobic_residues[:10]) if hydrophobic_residues else 'None detected'}
- Charged residues ({len(charged_residues)}): {', '.join(charged_residues[:10]) if charged_residues else 'None detected'}
- Aromatic residues ({len(aromatic_residues)}): {', '.join(aromatic_residues[:10]) if aromatic_residues else 'None detected'}

## Fragments to Analyze

"""
        
        for i, frag in enumerate(fragments, 1):
            smiles = frag.get('smiles', 'Unknown')
            
            # NCI counts for reference (not primary criterion)
            hb_count = frag.get('hbond_count', 0)
            pi_count = frag.get('pi_count', 0)
            hp_count = frag.get('hydrophobic_count', 0)
            sb_count = frag.get('salt_bridge_count', 0)
            
            # Structure score from Track 1 (for reference)
            structure_score = frag.get('structure_score', 'N/A')
            
            prompt += f"""
### Fragment {i}: `{smiles}`

**NCI Summary (REFERENCE ONLY - do not use as primary criterion):**
- H-bonds: {hb_count}, π-π: {pi_count}, Hydrophobic: {hp_count}, Salt bridges: {sb_count}
- Structure Score (Track 1): {structure_score}

"""
        
        prompt += """
## Analysis Criteria

Evaluate EACH fragment on these 6 dimensions based on CHEMICAL KNOWLEDGE:

### 1. Pharmacophore Quality (药效团质量) [0.0-1.0]
- Is this a recognized pharmacophore? (aromatic ring, H-bond donor/acceptor, charged group, etc.)
- How common is this pharmacophore in successful drugs?
- Score 1.0 = classical strong pharmacophore, 0.0 = no pharmacophore features

### 2. Electronic Properties (电子性质) [0.0-1.0]
- Electron density distribution (donating vs withdrawing groups)
- Polarizability and ability to form interactions
- Conjugation and resonance stabilization
- Score 1.0 = excellent electronic properties for binding

### 3. Pocket Complementarity (口袋互补性) [0.0-1.0]
- Does the fragment's polarity match the pocket region?
- Is the fragment appropriately sized for the pocket?
- Would it fill space effectively?
- Score 1.0 = excellent match to pocket characteristics

### 4. Drug-likeness (类药性) [0.0-1.0]
- Is this a "privileged scaffold" in medicinal chemistry?
- How often do similar fragments appear in approved drugs?
- Lipophilicity, solubility considerations
- Score 1.0 = highly drug-like fragment

### 5. Modifiability (可修饰性/SAR潜力) [0.0-1.0]
- Can this fragment be easily modified for optimization?
- Are there good vectors for SAR exploration?
- Synthetic accessibility
- Score 1.0 = highly modifiable with clear SAR vectors

### 6. Binding Anchor Potential (结合锚点潜力) [0.0-1.0]
- Could this fragment serve as a stable "anchor" in the binding site?
- Does it have features that would make it essential for binding?
- Based on chemical nature, not just detected interactions
- Score 1.0 = excellent anchor potential

## Output Format

Return ONLY valid JSON (no markdown, no explanation outside JSON):

```json
{
  "fragment_analysis": [
    {
      "fragment_id": 1,
      "smiles": "...",
      "scores": {
        "pharmacophore_quality": 0.0-1.0,
        "electronic_properties": 0.0-1.0,
        "pocket_complementarity": 0.0-1.0,
        "drug_likeness": 0.0-1.0,
        "modifiability": 0.0-1.0,
        "binding_anchor_potential": 0.0-1.0
      },
      "knowledge_score": 0.0-1.0,
      "key_insight": "One sentence explaining the chemical rationale",
      "confidence": "HIGH/MEDIUM/LOW"
    }
  ],
  "ranking_rationale": "Brief explanation of overall ranking logic"
}
```

**CRITICAL**: 
- Base analysis on CHEMICAL KNOWLEDGE, not NCI counts
- The knowledge_score should be a weighted average of the 6 dimensions
- Confidence should reflect how certain you are about the assessment
- Key insight should explain WHY this fragment is/isn't important from a chemistry perspective
"""
        
        return prompt
    
    def analyze_fragments(
        self,
        fragments: List[Dict],
        pocket_features: Dict,
        nci_summary: Optional[Dict] = None
    ) -> Dict[str, Dict]:
        """
        Analyze fragments using LLM chemical knowledge.
        
        Args:
            fragments: List of fragment data
            pocket_features: Pocket characteristics
            nci_summary: Optional NCI summary
            
        Returns:
            Dictionary mapping fragment SMILES to knowledge scores
        """
        print("   🧠 Building chemical knowledge prompt...")
        prompt = self.build_knowledge_prompt(fragments, pocket_features, nci_summary)
        
        print("   🤖 Calling LLM for chemical knowledge analysis...")
        messages = [
            {
                "role": "system",
                "content": "You are an expert medicinal chemist. Analyze fragments based on chemical knowledge, not just interaction counts. Always respond with valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = self._call_llm(messages, temperature=0.2, max_tokens=3000)
        
        if not response:
            print("   ⚠️  LLM call failed, returning empty results")
            return {}
        
        # Parse response
        return self._parse_knowledge_response(response, fragments)
    
    def _parse_knowledge_response(
        self,
        response: str,
        fragments: List[Dict]
    ) -> Dict[str, Dict]:
        """Parse LLM response and extract knowledge scores."""
        
        # Clean response
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"   ⚠️  Failed to parse LLM response: {e}")
            print(f"   Response preview: {cleaned[:200]}...")
            return self._generate_fallback_scores(fragments)
        
        # Extract scores
        knowledge_scores = {}
        
        fragment_analysis = result.get('fragment_analysis', [])
        
        for analysis in fragment_analysis:
            smiles = analysis.get('smiles', '')
            
            # Find matching fragment
            matching_frag = None
            for frag in fragments:
                if frag.get('smiles') == smiles:
                    matching_frag = frag
                    break
            
            if not matching_frag:
                # Try to match by fragment_id
                frag_id = analysis.get('fragment_id', 0) - 1
                if 0 <= frag_id < len(fragments):
                    matching_frag = fragments[frag_id]
                    smiles = matching_frag.get('smiles', smiles)
            
            scores = analysis.get('scores', {})
            
            # Calculate weighted knowledge score if not provided
            knowledge_score = analysis.get('knowledge_score')
            if knowledge_score is None:
                # Default weights for 6 dimensions
                weights = {
                    'pharmacophore_quality': 0.20,
                    'electronic_properties': 0.10,
                    'pocket_complementarity': 0.25,
                    'drug_likeness': 0.15,
                    'modifiability': 0.10,
                    'binding_anchor_potential': 0.20
                }
                
                knowledge_score = sum(
                    scores.get(k, 0.5) * w
                    for k, w in weights.items()
                )
            
            knowledge_scores[smiles] = {
                'knowledge_score': float(knowledge_score),
                'detailed_scores': scores,
                'key_insight': analysis.get('key_insight', ''),
                'confidence': analysis.get('confidence', 'MEDIUM')
            }
        
        return knowledge_scores
    
    def _generate_fallback_scores(
        self,
        fragments: List[Dict]
    ) -> Dict[str, Dict]:
        """Generate fallback scores if LLM fails."""
        print("   ⚠️  Using fallback scoring (LLM unavailable)")
        
        fallback_scores = {}
        
        for frag in fragments:
            smiles = frag.get('smiles', '')
            
            # Simple heuristic scoring based on SMILES
            score = 0.5  # Default middle score
            
            # Bonus for aromatic rings
            if 'c1' in smiles.lower() or 'C1' in smiles:
                score += 0.1
            
            # Bonus for H-bond capable groups
            if 'O' in smiles or 'N' in smiles:
                score += 0.1
            
            # Bonus for heterocycles
            if any(x in smiles for x in ['n', 'o', 's', 'N', 'O', 'S']):
                if '1' in smiles:  # Likely cyclic
                    score += 0.1
            
            score = min(score, 1.0)
            
            fallback_scores[smiles] = {
                'knowledge_score': score,
                'detailed_scores': {
                    'pharmacophore_quality': score,
                    'electronic_properties': 0.5,
                    'pocket_complementarity': 0.5,
                    'drug_likeness': score,
                    'modifiability': 0.5,
                    'binding_anchor_potential': score
                },
                'key_insight': 'Fallback scoring based on SMILES heuristics',
                'confidence': 'LOW'
            }
        
        return fallback_scores


class RuleBasedChemicalAnalyzer:
    """
    Rule-based fallback for Track 2 when LLM is not available.
    
    Uses chemical rules and heuristics instead of LLM.
    Less accurate but works without API.
    """
    
    def __init__(self):
        """Initialize rule-based analyzer."""
        # Known pharmacophore patterns
        self.pharmacophore_patterns = {
            'aromatic_ring': ['c1ccccc1', 'c1ccncc1', 'c1ccc2ccccc2c1'],
            'carboxylic_acid': ['C(=O)O', 'C(O)=O'],
            'amine': ['N', 'CN', 'CCN'],
            'amide': ['C(=O)N', 'NC=O'],
            'hydroxyl': ['O', 'CO'],
            'sulfone': ['S(=O)(=O)'],
            'heterocycle': ['n', 'o', 's']
        }
        
        # Privileged scaffolds in drug design
        self.privileged_scaffolds = [
            'benzene', 'pyridine', 'piperidine', 'morpholine',
            'indole', 'quinoline', 'benzimidazole'
        ]
    
    def analyze_fragments(
        self,
        fragments: List[Dict],
        pocket_features: Dict,
        nci_summary: Optional[Dict] = None
    ) -> Dict[str, Dict]:
        """
        Analyze fragments using chemical rules.
        
        Returns scores similar to LLM-based analysis.
        """
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
        
        knowledge_scores = {}
        
        for frag in fragments:
            smiles = frag.get('smiles', '')
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError("Invalid SMILES")
            except:
                knowledge_scores[smiles] = self._default_score()
                continue
            
            scores = {}
            
            # 1. Pharmacophore quality
            scores['pharmacophore_quality'] = self._score_pharmacophore(mol, smiles)
            
            # 2. Electronic properties
            scores['electronic_properties'] = self._score_electronic(mol)
            
            # 3. Pocket complementarity
            scores['pocket_complementarity'] = self._score_pocket_match(mol, pocket_features)
            
            # 4. Drug-likeness
            scores['drug_likeness'] = self._score_drug_likeness(mol)
            
            # 5. Modifiability
            scores['modifiability'] = self._score_modifiability(mol)
            
            # 6. Binding anchor potential
            scores['binding_anchor_potential'] = self._score_anchor_potential(mol, smiles)
            
            # Weighted average
            weights = {
                'pharmacophore_quality': 0.20,
                'electronic_properties': 0.10,
                'pocket_complementarity': 0.25,
                'drug_likeness': 0.15,
                'modifiability': 0.10,
                'binding_anchor_potential': 0.20
            }
            
            knowledge_score = sum(scores[k] * weights[k] for k in weights)
            
            # Determine confidence
            score_variance = sum((s - knowledge_score)**2 for s in scores.values()) / len(scores)
            if score_variance < 0.02 and knowledge_score > 0.6:
                confidence = 'HIGH'
            elif score_variance < 0.05:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
            
            knowledge_scores[smiles] = {
                'knowledge_score': knowledge_score,
                'detailed_scores': scores,
                'key_insight': self._generate_insight(mol, smiles, scores),
                'confidence': confidence
            }
        
        return knowledge_scores
    
    def _score_pharmacophore(self, mol, smiles: str) -> float:
        """Score pharmacophore quality."""
        from rdkit.Chem import Descriptors
        
        score = 0.3  # Base score
        
        # Check for aromatic rings
        if mol.GetAromaticAtoms():
            score += 0.2
        
        # Check for H-bond donors/acceptors
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        if hbd > 0:
            score += min(0.15, hbd * 0.05)
        if hba > 0:
            score += min(0.15, hba * 0.05)
        
        # Check for charged groups
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() != 0:
                score += 0.1
                break
        
        return min(score, 1.0)
    
    def _score_electronic(self, mol) -> float:
        """Score electronic properties."""
        from rdkit.Chem import Descriptors
        
        score = 0.5  # Base
        
        # Polarizability indicator
        tpsa = Descriptors.TPSA(mol)
        if 20 < tpsa < 140:  # Good range
            score += 0.2
        
        # Conjugation
        if mol.GetAromaticAtoms():
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_pocket_match(self, mol, pocket_features: Dict) -> float:
        """Score pocket complementarity."""
        from rdkit.Chem import Descriptors
        
        score = 0.5  # Base
        
        # Get pocket characteristics
        residues = pocket_features.get('residues', [])
        
        # Count pocket types
        polar_count = sum(1 for r in residues if str(r)[:3] in ['SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'])
        hydrophobic_count = sum(1 for r in residues if str(r)[:3] in ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP'])
        
        # Fragment characteristics
        tpsa = Descriptors.TPSA(mol)
        logp = Descriptors.MolLogP(mol)
        
        # Match polar pocket with polar fragment
        if polar_count > hydrophobic_count:
            if tpsa > 40:
                score += 0.2
        else:
            # Hydrophobic pocket prefers lipophilic fragments
            if logp > 1:
                score += 0.2
        
        return min(score, 1.0)
    
    def _score_drug_likeness(self, mol) -> float:
        """Score drug-likeness."""
        from rdkit.Chem import Descriptors
        
        score = 0.5
        
        # Simple fragment drug-likeness
        mw = Descriptors.MolWt(mol)
        if 50 < mw < 300:  # Fragment-like
            score += 0.2
        
        logp = Descriptors.MolLogP(mol)
        if -1 < logp < 4:
            score += 0.2
        
        # Aromatic bonus
        if mol.GetAromaticAtoms():
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_modifiability(self, mol) -> float:
        """Score modifiability/SAR potential."""
        score = 0.5
        
        # More atoms = more modifiable
        num_atoms = mol.GetNumAtoms()
        if num_atoms > 5:
            score += 0.2
        
        # Check for common modification points
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in ['N', 'O']:
                score += 0.1
                break
        
        return min(score, 1.0)
    
    def _score_anchor_potential(self, mol, smiles: str) -> float:
        """Score binding anchor potential."""
        from rdkit.Chem import Descriptors
        
        score = 0.4
        
        # Aromatic rings are good anchors
        if mol.GetAromaticAtoms():
            score += 0.2
        
        # Multiple interaction points
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        if hbd + hba >= 2:
            score += 0.2
        
        # Rigidity (rings)
        ring_info = mol.GetRingInfo()
        if ring_info.NumRings() > 0:
            score += 0.1
        
        return min(score, 1.0)
    
    def _generate_insight(self, mol, smiles: str, scores: Dict) -> str:
        """Generate insight text."""
        insights = []
        
        if scores['pharmacophore_quality'] > 0.7:
            insights.append("strong pharmacophore features")
        
        if scores['binding_anchor_potential'] > 0.7:
            insights.append("good anchor potential")
        
        if scores['drug_likeness'] > 0.7:
            insights.append("drug-like properties")
        
        if not insights:
            insights.append("moderate chemical properties")
        
        return f"Fragment shows {', '.join(insights)}"
    
    def _default_score(self) -> Dict:
        """Return default score for invalid fragments."""
        return {
            'knowledge_score': 0.3,
            'detailed_scores': {
                'pharmacophore_quality': 0.3,
                'electronic_properties': 0.3,
                'pocket_complementarity': 0.3,
                'drug_likeness': 0.3,
                'modifiability': 0.3,
                'binding_anchor_potential': 0.3
            },
            'key_insight': 'Unable to analyze fragment',
            'confidence': 'LOW'
        }


if __name__ == "__main__":
    print("✓ Chemical Knowledge Analyzer Module Loaded")
    print("\nTrack 2 Analysis Dimensions:")
    print("  1. Pharmacophore Quality (药效团质量)")
    print("  2. Electronic Properties (电子性质)")
    print("  3. Pocket Complementarity (口袋互补性)")
    print("  4. Drug-likeness (类药性)")
    print("  5. Modifiability (可修饰性)")
    print("  6. Binding Anchor Potential (结合锚点潜力)")
    print("\nUsage:")
    print("  # With LLM")
    print("  analyzer = ChemicalKnowledgeAnalyzer(api_key)")
    print("  scores = analyzer.analyze_fragments(fragments, pocket_features)")
    print("\n  # Without LLM (rule-based)")
    print("  analyzer = RuleBasedChemicalAnalyzer()")
    print("  scores = analyzer.analyze_fragments(fragments, pocket_features)")