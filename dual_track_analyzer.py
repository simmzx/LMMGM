"""
Dual-Track Fragment Analyzer for Diffusion Model Conditioning

This is the main integration module that combines:
- Track 1: Soft NCI Detection (structure-based scoring)
- Track 2: Chemical Knowledge Analysis (LLM-based scoring)

The dual-track approach provides more robust fragment selection by:
1. Softening NCI hard thresholds → preserves interaction quality information
2. Adding LLM chemical knowledge → captures chemistry not in 3D geometry
3. Fusing scores with confidence weighting → more reliable output

This is designed specifically for generating high-quality fragment conditions
for diffusion-based molecular generation models.

Author: For diffusion model conditioning
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem

# Import our modules
try:
    from soft_nci_detector import SoftNCIDetector, map_interactions_to_fragments_soft
    SOFT_NCI_AVAILABLE = True
except ImportError:
    print("⚠️  Soft NCI detector not available")
    SOFT_NCI_AVAILABLE = False

try:
    from chemical_knowledge_analyzer import ChemicalKnowledgeAnalyzer, RuleBasedChemicalAnalyzer
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    print("⚠️  Chemical knowledge analyzer not available")
    KNOWLEDGE_AVAILABLE = False

try:
    from llm_analyzer import FragmentAnalyzer
    from pocket_features import extract_pocket_features_from_pdb
    CORE_AVAILABLE = True
except ImportError:
    print("⚠️  Core modules not available")
    CORE_AVAILABLE = False


class ScoreFusionModule:
    """
    Fuses scores from Track 1 (structure) and Track 2 (knowledge).
    
    Fusion strategies:
    1. weighted_average: Simple weighted combination
    2. confidence_weighted: Weight by confidence levels
    3. geometric_mean: Both tracks must agree
    4. adaptive: Dynamically adjust based on score agreement
    """
    
    def __init__(self, strategy: str = 'confidence_weighted'):
        """
        Initialize fusion module.
        
        Args:
            strategy: Fusion strategy to use
        """
        self.strategy = strategy
        
        # Confidence to weight mapping
        self.confidence_weights = {
            'HIGH': 0.9,
            'MEDIUM': 0.6,
            'LOW': 0.3
        }
    
    def fuse_scores(
        self,
        structure_scores: Dict[str, Dict],
        knowledge_scores: Dict[str, Dict],
        fragment_list: List[str]
    ) -> Dict[str, Dict]:
        """
        Fuse scores from both tracks.
        
        Args:
            structure_scores: {smiles: {structure_score, confidence, ...}}
            knowledge_scores: {smiles: {knowledge_score, confidence, ...}}
            fragment_list: List of fragment SMILES
            
        Returns:
            Dictionary with fused scores for each fragment
        """
        fused_results = {}
        
        for frag_smiles in fragment_list:
            # Get scores from both tracks
            s_data = structure_scores.get(frag_smiles, {})
            k_data = knowledge_scores.get(frag_smiles, {})
            
            s_score = s_data.get('structure_score', 0.0)
            k_score = k_data.get('knowledge_score', 0.0)
            
            s_confidence = s_data.get('confidence', 'LOW')
            k_confidence = k_data.get('confidence', 'MEDIUM')
            
            # Calculate fused score based on strategy
            if self.strategy == 'weighted_average':
                # Fixed 60-40 weighting
                final_score = 0.6 * s_score + 0.4 * k_score
                
            elif self.strategy == 'confidence_weighted':
                # Weight by confidence
                s_weight = self.confidence_weights.get(s_confidence, 0.5)
                k_weight = self.confidence_weights.get(k_confidence, 0.5)
                
                total_weight = s_weight + k_weight
                if total_weight > 0:
                    final_score = (s_score * s_weight + k_score * k_weight) / total_weight
                else:
                    final_score = (s_score + k_score) / 2
                
            elif self.strategy == 'geometric_mean':
                # Both must be good
                if s_score > 0 and k_score > 0:
                    final_score = (s_score * k_score) ** 0.5
                else:
                    final_score = 0.0
                
            elif self.strategy == 'adaptive':
                # Adapt based on agreement
                agreement = 1 - abs(s_score - k_score)
                
                if agreement > 0.7:
                    # High agreement - use average
                    final_score = (s_score + k_score) / 2
                else:
                    # Disagreement - trust higher confidence
                    s_conf_val = self.confidence_weights.get(s_confidence, 0.5)
                    k_conf_val = self.confidence_weights.get(k_confidence, 0.5)
                    
                    if s_conf_val > k_conf_val:
                        final_score = 0.7 * s_score + 0.3 * k_score
                    else:
                        final_score = 0.3 * s_score + 0.7 * k_score
            else:
                # Default to simple average
                final_score = (s_score + k_score) / 2
            
            # Calculate score agreement
            score_agreement = 1 - abs(s_score - k_score)
            
            # Determine final confidence
            if score_agreement > 0.7:
                if s_confidence == 'HIGH' or k_confidence == 'HIGH':
                    final_confidence = 'HIGH'
                else:
                    final_confidence = 'MEDIUM'
            elif score_agreement > 0.4:
                final_confidence = 'MEDIUM'
            else:
                final_confidence = 'LOW'
            
            fused_results[frag_smiles] = {
                'final_score': final_score,
                'structure_score': s_score,
                'knowledge_score': k_score,
                'structure_confidence': s_confidence,
                'knowledge_confidence': k_confidence,
                'score_agreement': score_agreement,
                'final_confidence': final_confidence,
                'structure_details': s_data,
                'knowledge_details': k_data
            }
        
        return fused_results
    
    def rank_fragments(
        self,
        fused_results: Dict[str, Dict],
        min_confidence: str = 'LOW',
        min_score: float = 0.0
    ) -> List[Tuple[str, Dict]]:
        """
        Rank fragments by fused score.
        
        Args:
            fused_results: Output from fuse_scores()
            min_confidence: Minimum confidence to include
            min_score: Minimum score to include
            
        Returns:
            Sorted list of (smiles, data) tuples
        """
        confidence_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        min_conf_value = confidence_order.get(min_confidence, 1)
        
        # Filter
        filtered = {
            k: v for k, v in fused_results.items()
            if (confidence_order.get(v['final_confidence'], 0) >= min_conf_value
                and v['final_score'] >= min_score)
        }
        
        # Sort by confidence (primary) and score (secondary)
        ranked = sorted(
            filtered.items(),
            key=lambda x: (
                confidence_order.get(x[1]['final_confidence'], 0),
                x[1]['final_score']
            ),
            reverse=True
        )
        
        return ranked


class DualTrackFragmentAnalyzer:
    """
    Main class for dual-track fragment analysis.
    
    Combines:
    - Track 1: Soft NCI detection (structure-based)
    - Track 2: Chemical knowledge analysis (LLM-based)
    - Score fusion for robust fragment selection
    
    Designed for generating high-quality fragment conditions
    for diffusion-based molecular generation.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        fusion_strategy: str = 'confidence_weighted',
        use_llm_knowledge: bool = True
    ):
        """
        Initialize dual-track analyzer.
        
        Args:
            api_key: API key for LLM (optional if use_llm_knowledge=False)
            fusion_strategy: Score fusion strategy
            use_llm_knowledge: Whether to use LLM for Track 2 (False = rule-based)
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.use_llm_knowledge = use_llm_knowledge
        
        # Initialize fusion module
        self.fusion_module = ScoreFusionModule(fusion_strategy)
        
        # Initialize Track 2 analyzer
        if use_llm_knowledge and self.api_key:
            self.knowledge_analyzer = ChemicalKnowledgeAnalyzer(self.api_key)
        else:
            self.knowledge_analyzer = RuleBasedChemicalAnalyzer()
            if use_llm_knowledge:
                print("⚠️  No API key provided, using rule-based knowledge analysis")
    
    def analyze(
        self,
        ligand_mol: Chem.Mol,
        protein_pdb: str,
        pocket_features: Dict,
        save_coordinates: bool = True,
        output_dir: str = 'diffusion_input'
    ) -> Dict:
        """
        Run complete dual-track analysis.
        
        Args:
            ligand_mol: RDKit molecule with 3D coordinates
            protein_pdb: Path to protein PDB file
            pocket_features: Pocket characteristics dictionary
            save_coordinates: Whether to save coordinate files
            output_dir: Directory for output files
            
        Returns:
            Comprehensive analysis results dictionary
        """
        print("\n" + "="*70)
        print("🔬 DUAL-TRACK FRAGMENT ANALYSIS FOR DIFFUSION MODEL")
        print("="*70)
        
        # Step 1: Fragment decomposition
        print("\n[Step 1] 📦 BRICS Fragment Decomposition...")
        frags, labels = self._decompose_ligand(ligand_mol)
        print(f"   ✓ Found {len(frags)} fragments")
        for i, frag in enumerate(frags):
            print(f"      Fragment {i+1}: {frag}")
        
        # Step 2: Track 1 - Soft NCI Detection
        print("\n[Step 2] 🔍 Track 1: Soft NCI Detection...")
        structure_scores = self._run_track1(ligand_mol, protein_pdb, frags, labels)
        
        print("   Results:")
        for frag, data in structure_scores.items():
            score = data.get('structure_score', 0)
            conf = data.get('confidence', 'N/A')
            print(f"      {frag[:40]:40s} → S_score: {score:.3f} ({conf})")
        
        # Step 3: Track 2 - Chemical Knowledge Analysis
        print("\n[Step 3] 🧠 Track 2: Chemical Knowledge Analysis...")
        
        # Prepare fragment data for Track 2
        frag_data = self._prepare_frag_data(frags, structure_scores)
        
        knowledge_scores = self.knowledge_analyzer.analyze_fragments(
            frag_data, pocket_features
        )
        
        print("   Results:")
        for frag, data in knowledge_scores.items():
            score = data.get('knowledge_score', 0)
            conf = data.get('confidence', 'N/A')
            insight = data.get('key_insight', '')[:50]
            print(f"      {frag[:40]:40s} → K_score: {score:.3f} ({conf})")
            if insight:
                print(f"         💡 {insight}...")
        
        # Step 4: Score Fusion
        print("\n[Step 4] 🔗 Score Fusion...")
        fused_results = self.fusion_module.fuse_scores(
            structure_scores, knowledge_scores, frags
        )
        
        # Step 5: Ranking
        print("\n[Step 5] 📊 Ranking Fragments...")
        ranked_fragments = self.fusion_module.rank_fragments(fused_results)
        
        # Display results
        print("\n" + "="*70)
        print("🎯 CRITICAL FRAGMENTS FOR DIFFUSION MODEL")
        print("="*70)
        
        critical_fragments = []
        for rank, (frag_smiles, data) in enumerate(ranked_fragments[:5], 1):
            print(f"\n🏆 Rank {rank}: {frag_smiles}")
            print(f"   ├─ Final Score: {data['final_score']:.3f}")
            print(f"   ├─ Structure Score (Track 1): {data['structure_score']:.3f} ({data['structure_confidence']})")
            print(f"   ├─ Knowledge Score (Track 2): {data['knowledge_score']:.3f} ({data['knowledge_confidence']})")
            print(f"   ├─ Score Agreement: {data['score_agreement']:.3f}")
            print(f"   └─ Final Confidence: {data['final_confidence']}")
            
            # Add insight if available
            k_details = data.get('knowledge_details', {})
            if k_details.get('key_insight'):
                print(f"   💡 Insight: {k_details['key_insight']}")
            
            critical_fragments.append({
                'rank': rank,
                'fragment_smiles': frag_smiles,
                'final_score': data['final_score'],
                'structure_score': data['structure_score'],
                'knowledge_score': data['knowledge_score'],
                'score_agreement': data['score_agreement'],
                'confidence': data['final_confidence'],
                'rationale': self._generate_rationale(frag_smiles, data)
            })
        
        # Step 6: Extract coordinates
        print("\n[Step 6] 📍 Extracting Coordinates...")
        fragment_coords = self._extract_coordinates(ligand_mol, labels, frags)
        
        # Step 7: Save files (if requested)
        saved_files = {}
        if save_coordinates:
            print(f"\n[Step 7] 💾 Saving Files to {output_dir}/...")
            saved_files = self._save_files(
                ligand_mol, protein_pdb, frags, labels,
                fragment_coords, critical_fragments, output_dir
            )
        
        # Compile final results
        results = {
            'all_fragments': frags,
            'fragment_labels': labels,
            'structure_scores': structure_scores,
            'knowledge_scores': knowledge_scores,
            'fused_results': fused_results,
            'critical_fragments': critical_fragments,
            'fragment_coordinates': fragment_coords,
            'saved_files': saved_files,
            'analysis_metadata': {
                'fusion_strategy': self.fusion_module.strategy,
                'use_llm_knowledge': self.use_llm_knowledge,
                'track1_type': 'Soft NCI',
                'track2_type': 'LLM Chemical Knowledge' if self.use_llm_knowledge else 'Rule-based'
            }
        }
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _decompose_ligand(self, ligand_mol: Chem.Mol) -> Tuple[List[str], List[int]]:
        """Decompose ligand using BRICS."""
        from rdkit.Chem import BRICS
        
        try:
            bonds = list(BRICS.FindBRICSBonds(ligand_mol))
            
            if bonds:
                frags_mol = BRICS.BreakBRICSBonds(ligand_mol)
                frags_list = Chem.GetMolFrags(frags_mol, asMols=True)
                frags = [Chem.MolToSmiles(f) for f in frags_list]
                
                # Create labeling
                labels = [0] * ligand_mol.GetNumAtoms()
                atom_idx = 0
                for frag_idx, frag_mol in enumerate(frags_list):
                    for _ in range(frag_mol.GetNumAtoms()):
                        if atom_idx < len(labels):
                            labels[atom_idx] = frag_idx
                            atom_idx += 1
            else:
                frags = [Chem.MolToSmiles(ligand_mol)]
                labels = [0] * ligand_mol.GetNumAtoms()
            
            return frags, labels
            
        except Exception as e:
            print(f"   ⚠️  BRICS failed: {e}")
            return [Chem.MolToSmiles(ligand_mol)], [0] * ligand_mol.GetNumAtoms()
    
    def _run_track1(
        self,
        ligand_mol: Chem.Mol,
        protein_pdb: str,
        frags: List[str],
        labels: List[int]
    ) -> Dict[str, Dict]:
        """Run Track 1: Soft NCI detection."""
        
        # Initialize soft NCI detector
        soft_detector = SoftNCIDetector(protein_pdb)
        
        # Detect all interactions with soft scoring
        interactions = soft_detector.detect_all_soft(ligand_mol)
        
        # Print detection summary
        print(f"   Detected interactions (soft threshold):")
        for int_type, int_list in interactions.items():
            strong = sum(1 for i in int_list if i.get('strength') == 'STRONG' or i.get('combined_score', i.get('score', 0)) > 0.7)
            print(f"      {int_type}: {len(int_list)} total ({strong} strong)")
        
        # Map to fragments
        fragment_interactions = map_interactions_to_fragments_soft(
            interactions, labels, frags
        )
        
        # Calculate structure scores for each fragment
        structure_scores = {}
        for frag_smiles in frags:
            frag_int = fragment_interactions.get(frag_smiles, {
                'hydrogen_bonds': [],
                'pi_stacking': [],
                'hydrophobic': [],
                'salt_bridges': []
            })
            
            score_data = soft_detector.calculate_fragment_structure_score(frag_int)
            structure_scores[frag_smiles] = score_data
        
        return structure_scores
    
    def _prepare_frag_data(
        self,
        frags: List[str],
        structure_scores: Dict[str, Dict]
    ) -> List[Dict]:
        """Prepare fragment data for Track 2 analysis."""
        frag_data = []
        
        for frag_smiles in frags:
            s_data = structure_scores.get(frag_smiles, {})
            breakdown = s_data.get('breakdown', {})
            
            frag_data.append({
                'smiles': frag_smiles,
                'hbond_count': breakdown.get('hydrogen_bonds', {}).get('count', 0),
                'pi_count': breakdown.get('pi_stacking', {}).get('count', 0),
                'hydrophobic_count': breakdown.get('hydrophobic', {}).get('count', 0),
                'salt_bridge_count': breakdown.get('salt_bridges', {}).get('count', 0),
                'structure_score': s_data.get('structure_score', 0)
            })
        
        return frag_data
    
    def _extract_coordinates(
        self,
        ligand_mol: Chem.Mol,
        labels: List[int],
        frags: List[str]
    ) -> Dict[str, Dict]:
        """Extract 3D coordinates for each fragment."""
        if ligand_mol.GetNumConformers() == 0:
            return {}
        
        conf = ligand_mol.GetConformer()
        fragment_coords = {}
        
        for frag_idx, frag_smiles in enumerate(frags):
            atom_indices = [i for i, label in enumerate(labels) if label == frag_idx]
            
            if not atom_indices:
                continue
            
            coords = []
            for atom_idx in atom_indices:
                pos = conf.GetAtomPosition(atom_idx)
                coords.append([pos.x, pos.y, pos.z])
            
            coords_array = np.array(coords)
            centroid = coords_array.mean(axis=0)
            
            fragment_coords[frag_smiles] = {
                'atom_indices': atom_indices,
                'coordinates': coords_array.tolist(),
                'centroid': centroid.tolist(),
                'num_atoms': len(atom_indices)
            }
        
        return fragment_coords
    
    def _generate_rationale(self, frag_smiles: str, data: Dict) -> str:
        """Generate human-readable rationale."""
        parts = []
        
        # Structure analysis
        s_score = data.get('structure_score', 0)
        if s_score > 0.7:
            parts.append("strong structural interactions detected")
        elif s_score > 0.4:
            parts.append("moderate structural interactions")
        
        # Knowledge analysis
        k_details = data.get('knowledge_details', {})
        if k_details.get('key_insight'):
            parts.append(k_details['key_insight'])
        
        # Agreement
        agreement = data.get('score_agreement', 0)
        if agreement > 0.8:
            parts.append("high agreement between structural and chemical analysis")
        
        # Confidence
        conf = data.get('final_confidence', 'MEDIUM')
        if conf == 'HIGH':
            parts.append("high confidence fragment for diffusion conditioning")
        
        return '; '.join(parts) if parts else "Fragment identified through dual-track analysis"
    
    def _save_files(
        self,
        ligand_mol: Chem.Mol,
        protein_pdb: str,
        frags: List[str],
        labels: List[int],
        fragment_coords: Dict,
        critical_fragments: List[Dict],
        output_dir: str
    ) -> Dict:
        """Save coordinate files for diffusion model."""
        import os
        import shutil
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {'fragments': [], 'critical': []}
        
        # Save full ligand
        ligand_file = os.path.join(output_dir, 'ligand_full.sdf')
        try:
            writer = Chem.SDWriter(ligand_file)
            writer.write(ligand_mol)
            writer.close()
            saved_files['ligand'] = ligand_file
            print(f"   ✓ Saved ligand: {ligand_file}")
        except Exception as e:
            print(f"   ⚠️  Failed to save ligand: {e}")
        
        # Save protein pocket
        pocket_file = os.path.join(output_dir, 'pocket.pdb')
        try:
            shutil.copy(protein_pdb, pocket_file)
            saved_files['pocket'] = pocket_file
            print(f"   ✓ Saved pocket: {pocket_file}")
        except Exception as e:
            print(f"   ⚠️  Failed to save pocket: {e}")
        
        # Save critical fragments
        for crit_frag in critical_fragments[:3]:
            rank = crit_frag['rank']
            frag_smiles = crit_frag['fragment_smiles']
            
            if frag_smiles in fragment_coords:
                frag_file = os.path.join(output_dir, f'critical_fragment_rank{rank}.sdf')
                success = self._save_fragment_sdf(
                    ligand_mol,
                    fragment_coords[frag_smiles]['atom_indices'],
                    frag_file
                )
                
                if success:
                    saved_files['critical'].append({
                        'rank': rank,
                        'file': frag_file,
                        'smiles': frag_smiles
                    })
                    print(f"   ✓ Saved critical fragment (Rank {rank}): {frag_file}")
        
        # Save coordinates JSON
        json_file = os.path.join(output_dir, 'coordinates.json')
        coord_data = {
            'critical_fragments': critical_fragments,
            'fragment_coordinates': fragment_coords,
            'files': saved_files
        }
        
        with open(json_file, 'w') as f:
            json.dump(coord_data, f, indent=2)
        
        saved_files['json'] = json_file
        print(f"   ✓ Saved coordinates JSON: {json_file}")
        
        return saved_files
    
    def _save_fragment_sdf(
        self,
        ligand_mol: Chem.Mol,
        atom_indices: List[int],
        output_file: str
    ) -> bool:
        """Save a fragment as SDF file."""
        try:
            emol = Chem.EditableMol(Chem.Mol())
            conf = ligand_mol.GetConformer()
            
            old_to_new = {}
            for new_idx, old_idx in enumerate(atom_indices):
                atom = ligand_mol.GetAtomWithIdx(old_idx)
                emol.AddAtom(atom)
                old_to_new[old_idx] = new_idx
            
            for bond in ligand_mol.GetBonds():
                begin = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                if begin in atom_indices and end in atom_indices:
                    emol.AddBond(old_to_new[begin], old_to_new[end], bond.GetBondType())
            
            frag_mol = emol.GetMol()
            
            try:
                Chem.SanitizeMol(frag_mol)
            except:
                Chem.SanitizeMol(frag_mol, 
                    sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            
            frag_conf = Chem.Conformer(len(atom_indices))
            for new_idx, old_idx in enumerate(atom_indices):
                pos = conf.GetAtomPosition(old_idx)
                frag_conf.SetAtomPosition(new_idx, pos)
            
            frag_mol.AddConformer(frag_conf)
            
            writer = Chem.SDWriter(output_file)
            writer.SetKekulize(False)
            writer.write(frag_mol)
            writer.close()
            
            return True
            
        except Exception as e:
            print(f"   ⚠️  Failed to save fragment: {e}")
            return False
    
    def _print_summary(self, results: Dict):
        """Print analysis summary."""
        print("\n" + "="*70)
        print("📋 ANALYSIS SUMMARY")
        print("="*70)
        
        meta = results.get('analysis_metadata', {})
        print(f"\n📊 Configuration:")
        print(f"   Track 1: {meta.get('track1_type', 'N/A')}")
        print(f"   Track 2: {meta.get('track2_type', 'N/A')}")
        print(f"   Fusion Strategy: {meta.get('fusion_strategy', 'N/A')}")
        
        print(f"\n🎯 Top Critical Fragments for Diffusion Model:")
        for crit in results.get('critical_fragments', [])[:3]:
            print(f"   Rank {crit['rank']}: {crit['fragment_smiles']}")
            print(f"      Score: {crit['final_score']:.3f} | Confidence: {crit['confidence']}")
        
        saved = results.get('saved_files', {})
        if saved:
            print(f"\n💾 Output Files:")
            if saved.get('ligand'):
                print(f"   Ligand: {saved['ligand']}")
            if saved.get('pocket'):
                print(f"   Pocket: {saved['pocket']}")
            for crit_file in saved.get('critical', []):
                print(f"   Critical Fragment Rank {crit_file['rank']}: {crit_file['file']}")
            if saved.get('json'):
                print(f"   Coordinates JSON: {saved['json']}")
        
        print("\n" + "="*70)
        print("✅ Dual-Track Analysis Complete!")
        print("="*70)


if __name__ == "__main__":
    print("✓ Dual-Track Fragment Analyzer Module Loaded")
    print("\nArchitecture:")
    print("  Track 1: Soft NCI Detection (structure-based)")
    print("  Track 2: Chemical Knowledge Analysis (LLM/rule-based)")
    print("  Fusion: Confidence-weighted score combination")
    print("\nUsage:")
    print("  analyzer = DualTrackFragmentAnalyzer(api_key='...')")
    print("  results = analyzer.analyze(ligand_mol, protein_pdb, pocket_features)")
    print("\nFor diffusion model input, use:")
    print("  - diffusion_input/critical_fragment_rank1.sdf")
    print("  - diffusion_input/pocket.pdb")
    print("  - diffusion_input/coordinates.json")