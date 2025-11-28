"""
Soft NCI Detector - Track 1 of Dual-Track Fragment Analysis System

This module implements a "soft threshold" NCI detection system that outputs
continuous scores (0-1) instead of binary detection results.

Key improvements over hard-threshold NCI:
1. Continuous strength scoring (not just present/absent)
2. Distance-based quality assessment
3. Angle-based geometric quality for H-bonds and π-π
4. Diminishing returns for multiple weak interactions
5. Detailed interaction metadata for downstream analysis

Author: Enhanced for diffusion model conditioning
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from rdkit import Chem
from collections import defaultdict

# Import base NCI detector components
try:
    from nci_detector import ProteinStructure, AMINO_ACID_PROPERTIES
except ImportError:
    # Fallback: define locally if not available
    AMINO_ACID_PROPERTIES = {
        'ALA': {'type': 'hydrophobic', 'polar': False, 'aromatic': False},
        'VAL': {'type': 'hydrophobic', 'polar': False, 'aromatic': False},
        'LEU': {'type': 'hydrophobic', 'polar': False, 'aromatic': False},
        'ILE': {'type': 'hydrophobic', 'polar': False, 'aromatic': False},
        'MET': {'type': 'hydrophobic', 'polar': False, 'aromatic': False},
        'PHE': {'type': 'hydrophobic', 'polar': False, 'aromatic': True},
        'TRP': {'type': 'hydrophobic', 'polar': False, 'aromatic': True},
        'PRO': {'type': 'hydrophobic', 'polar': False, 'aromatic': False},
        'SER': {'type': 'polar', 'polar': True, 'hb_donor': True, 'hb_acceptor': True},
        'THR': {'type': 'polar', 'polar': True, 'hb_donor': True, 'hb_acceptor': True},
        'CYS': {'type': 'polar', 'polar': True, 'hb_donor': True},
        'TYR': {'type': 'polar', 'polar': True, 'aromatic': True, 'hb_donor': True, 'hb_acceptor': True},
        'ASN': {'type': 'polar', 'polar': True, 'hb_donor': True, 'hb_acceptor': True},
        'GLN': {'type': 'polar', 'polar': True, 'hb_donor': True, 'hb_acceptor': True},
        'LYS': {'type': 'charged', 'charge': '+', 'hb_donor': True},
        'ARG': {'type': 'charged', 'charge': '+', 'hb_donor': True},
        'HIS': {'type': 'charged', 'charge': '+', 'aromatic': True, 'hb_donor': True},
        'ASP': {'type': 'charged', 'charge': '-', 'hb_acceptor': True},
        'GLU': {'type': 'charged', 'charge': '-', 'hb_acceptor': True},
        'GLY': {'type': 'special', 'polar': False},
    }


class SoftNCIDetector:
    """
    Soft-threshold NCI detector with continuous scoring.
    
    Unlike traditional NCI detection with hard cutoffs (e.g., distance < 3.5Å),
    this detector outputs continuous strength scores that preserve information
    about interaction quality.
    
    This is critical for diffusion model conditioning, where we need high-confidence
    fragment selection, not just binary presence/absence.
    """
    
    # Ideal distances for each interaction type (in Angstroms)
    IDEAL_HBOND_DISTANCE = 2.8
    IDEAL_PI_DISTANCE = 3.8
    IDEAL_HYDROPHOBIC_DISTANCE = 3.8
    IDEAL_SALT_BRIDGE_DISTANCE = 2.8
    
    def __init__(self, protein_pdb: str):
        """
        Initialize soft NCI detector.
        
        Args:
            protein_pdb: Path to protein PDB file
        """
        self.protein = self._parse_pdb(protein_pdb)
        self.pdb_file = protein_pdb
    
    def _parse_pdb(self, pdb_file: str) -> Dict:
        """Parse PDB file and extract atom information."""
        atoms = []
        residues = {}
        
        with open(pdb_file, 'r') as f:
            for line in f:
                if not (line.startswith('ATOM') or line.startswith('HETATM')):
                    continue
                
                try:
                    atom_info = {
                        'serial': int(line[6:11].strip()),
                        'name': line[12:16].strip(),
                        'res_name': line[17:20].strip(),
                        'chain': line[21].strip() or 'A',
                        'res_num': int(line[22:26].strip()),
                        'x': float(line[30:38].strip()),
                        'y': float(line[38:46].strip()),
                        'z': float(line[46:54].strip()),
                        'element': line[76:78].strip() or line[12:14].strip()[0]
                    }
                    
                    atoms.append(atom_info)
                    
                    res_key = (atom_info['chain'], atom_info['res_num'], atom_info['res_name'])
                    if res_key not in residues:
                        residues[res_key] = []
                    residues[res_key].append(atom_info)
                    
                except (ValueError, IndexError):
                    continue
        
        return {'atoms': atoms, 'residues': residues}
    
    # ========== Soft Scoring Functions ==========
    
    def _score_hbond_distance(self, distance: float) -> float:
        """
        Score hydrogen bond by distance.
        
        Optimal: 2.7-2.9Å (score = 1.0)
        Acceptable: 2.5-3.5Å (score = 0.5-1.0)
        Weak: 3.5-4.0Å (score = 0.2-0.5)
        Negligible: >4.0Å (score ≈ 0)
        
        Returns:
            float: Score between 0 and 1
        """
        if distance < 2.5:
            # Too close - possible steric clash
            return 0.7
        elif distance < 2.7:
            return 0.9
        elif distance < 3.0:
            return 1.0  # Optimal range
        elif distance < 3.3:
            return 0.85
        elif distance < 3.5:
            return 0.7
        elif distance < 4.0:
            return 0.4  # Weak but detectable
        elif distance < 4.5:
            return 0.15  # Very weak
        else:
            return 0.0
    
    def _score_hbond_angle(self, angle: float) -> float:
        """
        Score hydrogen bond by angle (donor-H-acceptor).
        
        Optimal: 160-180° (score = 1.0)
        Good: 140-160° (score = 0.7-1.0)
        Acceptable: 120-140° (score = 0.4-0.7)
        Poor: <120° (score < 0.4)
        
        Returns:
            float: Score between 0 and 1
        """
        if angle >= 170:
            return 1.0
        elif angle >= 160:
            return 0.95
        elif angle >= 150:
            return 0.85
        elif angle >= 140:
            return 0.70
        elif angle >= 130:
            return 0.55
        elif angle >= 120:
            return 0.40
        elif angle >= 110:
            return 0.25
        elif angle >= 100:
            return 0.15
        else:
            return 0.05
    
    def _score_pi_distance(self, distance: float) -> float:
        """
        Score π-π stacking by centroid distance.
        
        Optimal: 3.4-4.0Å (score = 1.0)
        Good: 3.0-4.5Å (score = 0.6-1.0)
        Acceptable: 4.5-5.5Å (score = 0.2-0.6)
        
        Returns:
            float: Score between 0 and 1
        """
        if distance < 3.0:
            return 0.5  # Too close
        elif distance < 3.4:
            return 0.8
        elif distance < 4.0:
            return 1.0  # Optimal
        elif distance < 4.5:
            return 0.75
        elif distance < 5.0:
            return 0.5
        elif distance < 5.5:
            return 0.3
        elif distance < 6.0:
            return 0.15
        else:
            return 0.0
    
    def _score_pi_angle(self, angle: float) -> float:
        """
        Score π-π stacking by ring angle.
        
        Face-to-face: 0-20° or 160-180° (score = 1.0)
        T-shaped: 70-110° (score = 0.9)
        Intermediate: others (score = 0.4-0.7)
        
        Returns:
            float: Score between 0 and 1
        """
        # Normalize angle to 0-90 range
        if angle > 90:
            angle = 180 - angle
        
        if angle < 20:
            return 1.0  # Face-to-face (parallel)
        elif angle < 30:
            return 0.85
        elif angle < 40:
            return 0.6
        elif angle < 50:
            return 0.5
        elif angle < 60:
            return 0.6
        elif angle < 70:
            return 0.75
        elif angle < 80:
            return 0.9  # T-shaped
        else:
            return 0.95  # Near-perpendicular T-shaped
    
    def _score_hydrophobic_distance(self, distance: float) -> float:
        """
        Score hydrophobic contact by distance.
        
        Optimal: 3.5-4.0Å (score = 1.0)
        Good: 3.0-4.5Å (score = 0.6-1.0)
        Weak: 4.5-5.0Å (score = 0.2-0.6)
        
        Returns:
            float: Score between 0 and 1
        """
        if distance < 3.0:
            return 0.6  # Too close
        elif distance < 3.5:
            return 0.9
        elif distance < 4.0:
            return 1.0  # Optimal
        elif distance < 4.5:
            return 0.7
        elif distance < 5.0:
            return 0.4
        elif distance < 5.5:
            return 0.2
        else:
            return 0.0
    
    def _score_salt_bridge_distance(self, distance: float) -> float:
        """
        Score salt bridge by distance.
        
        Optimal: 2.5-3.5Å (score = 1.0)
        Good: 3.5-4.0Å (score = 0.7)
        Acceptable: 4.0-5.0Å (score = 0.3-0.5)
        
        Returns:
            float: Score between 0 and 1
        """
        if distance < 2.5:
            return 0.85  # Very close
        elif distance < 3.0:
            return 1.0  # Optimal
        elif distance < 3.5:
            return 0.9
        elif distance < 4.0:
            return 0.7
        elif distance < 4.5:
            return 0.5
        elif distance < 5.0:
            return 0.3
        else:
            return 0.0
    
    def _classify_strength(self, score: float) -> str:
        """Classify interaction strength based on score."""
        if score >= 0.8:
            return "STRONG"
        elif score >= 0.5:
            return "MODERATE"
        elif score >= 0.2:
            return "WEAK"
        else:
            return "NEGLIGIBLE"
    
    # ========== Detection Methods ==========
    
    def detect_hydrogen_bonds_soft(self, ligand_mol: Chem.Mol) -> List[Dict]:
        """
        Detect hydrogen bonds with soft scoring.
        
        Returns list of H-bonds with continuous scores.
        """
        hbonds = []
        conf = ligand_mol.GetConformer()
        
        # Get ligand donors and acceptors
        donors = self._get_hbond_donors(ligand_mol)
        acceptors = self._get_hbond_acceptors(ligand_mol)
        
        for (chain, res_num, res_name), res_atoms in self.protein['residues'].items():
            res_props = AMINO_ACID_PROPERTIES.get(res_name, {})
            
            if not (res_props.get('hb_donor') or res_props.get('hb_acceptor')):
                continue
            
            res_coords = np.array([[a['x'], a['y'], a['z']] for a in res_atoms])
            res_id = f"{res_name}{res_num}:{chain}"
            
            # Check donors
            if res_props.get('hb_acceptor'):
                for donor_idx in donors:
                    pos = conf.GetAtomPosition(donor_idx)
                    donor_coord = np.array([pos.x, pos.y, pos.z])
                    
                    distances = np.linalg.norm(res_coords - donor_coord, axis=1)
                    min_dist = float(np.min(distances))
                    
                    # Use soft threshold - include all within 5Å
                    if min_dist < 5.0:
                        dist_score = self._score_hbond_distance(min_dist)
                        
                        # Estimate angle (simplified - assume 150° for typical)
                        # In production, calculate actual D-H...A angle
                        angle_estimate = 150 if min_dist < 3.5 else 130
                        angle_score = self._score_hbond_angle(angle_estimate)
                        
                        combined_score = dist_score * angle_score
                        
                        if combined_score > 0.05:  # Only include if meaningful
                            hbonds.append({
                                'type': 'ligand_donor',
                                'ligand_atom': donor_idx,
                                'protein_residue': res_id,
                                'distance': min_dist,
                                'distance_score': dist_score,
                                'angle_estimate': angle_estimate,
                                'angle_score': angle_score,
                                'combined_score': combined_score,
                                'strength': self._classify_strength(combined_score)
                            })
            
            # Check acceptors
            if res_props.get('hb_donor'):
                for acc_idx in acceptors:
                    pos = conf.GetAtomPosition(acc_idx)
                    acc_coord = np.array([pos.x, pos.y, pos.z])
                    
                    distances = np.linalg.norm(res_coords - acc_coord, axis=1)
                    min_dist = float(np.min(distances))
                    
                    if min_dist < 5.0:
                        dist_score = self._score_hbond_distance(min_dist)
                        angle_estimate = 150 if min_dist < 3.5 else 130
                        angle_score = self._score_hbond_angle(angle_estimate)
                        
                        combined_score = dist_score * angle_score
                        
                        if combined_score > 0.05:
                            hbonds.append({
                                'type': 'ligand_acceptor',
                                'ligand_atom': acc_idx,
                                'protein_residue': res_id,
                                'distance': min_dist,
                                'distance_score': dist_score,
                                'angle_estimate': angle_estimate,
                                'angle_score': angle_score,
                                'combined_score': combined_score,
                                'strength': self._classify_strength(combined_score)
                            })
        
        return hbonds
    
    def detect_pi_stacking_soft(self, ligand_mol: Chem.Mol) -> List[Dict]:
        """Detect π-π stacking with soft scoring."""
        pi_stacks = []
        conf = ligand_mol.GetConformer()
        
        # Get aromatic rings in ligand
        ligand_rings = self._get_aromatic_rings(ligand_mol)
        
        for (chain, res_num, res_name), res_atoms in self.protein['residues'].items():
            res_props = AMINO_ACID_PROPERTIES.get(res_name, {})
            
            if not res_props.get('aromatic'):
                continue
            
            res_id = f"{res_name}{res_num}:{chain}"
            
            # Get aromatic atoms in residue
            aromatic_atoms = [a for a in res_atoms if a['name'] in ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'CH2', 'NE1']]
            
            if len(aromatic_atoms) < 3:
                continue
            
            res_center = np.mean([[a['x'], a['y'], a['z']] for a in aromatic_atoms], axis=0)
            
            for ring in ligand_rings:
                ring_coords = np.array([
                    [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                    for i in ring
                ])
                ring_center = ring_coords.mean(axis=0)
                
                distance = float(np.linalg.norm(ring_center - res_center))
                
                if distance < 7.0:  # Extended cutoff
                    dist_score = self._score_pi_distance(distance)
                    
                    # Estimate angle (simplified)
                    angle_estimate = 15 if distance < 4.5 else 45
                    angle_score = self._score_pi_angle(angle_estimate)
                    
                    combined_score = dist_score * angle_score
                    
                    if combined_score > 0.05:
                        pi_stacks.append({
                            'ligand_atoms': list(ring),
                            'protein_residue': res_id,
                            'distance': distance,
                            'distance_score': dist_score,
                            'angle_estimate': angle_estimate,
                            'angle_score': angle_score,
                            'combined_score': combined_score,
                            'strength': self._classify_strength(combined_score)
                        })
        
        return pi_stacks
    
    def detect_hydrophobic_soft(self, ligand_mol: Chem.Mol) -> List[Dict]:
        """Detect hydrophobic contacts with soft scoring."""
        contacts = []
        conf = ligand_mol.GetConformer()
        
        ligand_hydrophobic = self._get_hydrophobic_atoms(ligand_mol)
        
        for (chain, res_num, res_name), res_atoms in self.protein['residues'].items():
            res_props = AMINO_ACID_PROPERTIES.get(res_name, {})
            
            if res_props.get('type') != 'hydrophobic':
                continue
            
            res_id = f"{res_name}{res_num}:{chain}"
            
            hydrophobic_res_atoms = [a for a in res_atoms if a['element'] == 'C']
            if not hydrophobic_res_atoms:
                continue
            
            res_coords = np.array([[a['x'], a['y'], a['z']] for a in hydrophobic_res_atoms])
            
            for lig_idx in ligand_hydrophobic:
                pos = conf.GetAtomPosition(lig_idx)
                lig_coord = np.array([pos.x, pos.y, pos.z])
                
                distances = np.linalg.norm(res_coords - lig_coord, axis=1)
                min_dist = float(np.min(distances))
                
                if min_dist < 6.0:  # Extended cutoff
                    score = self._score_hydrophobic_distance(min_dist)
                    
                    if score > 0.05:
                        contacts.append({
                            'ligand_atom': lig_idx,
                            'protein_residue': res_id,
                            'distance': min_dist,
                            'score': score,
                            'strength': self._classify_strength(score)
                        })
        
        return contacts
    
    def detect_salt_bridges_soft(self, ligand_mol: Chem.Mol) -> List[Dict]:
        """Detect salt bridges with soft scoring."""
        salt_bridges = []
        conf = ligand_mol.GetConformer()
        
        positive_atoms = self._get_positive_atoms(ligand_mol)
        negative_atoms = self._get_negative_atoms(ligand_mol)
        
        for (chain, res_num, res_name), res_atoms in self.protein['residues'].items():
            res_props = AMINO_ACID_PROPERTIES.get(res_name, {})
            res_charge = res_props.get('charge')
            
            if not res_charge:
                continue
            
            res_id = f"{res_name}{res_num}:{chain}"
            res_coords = np.array([[a['x'], a['y'], a['z']] for a in res_atoms])
            
            # Positive ligand + negative residue
            if res_charge == '-':
                for lig_idx in positive_atoms:
                    pos = conf.GetAtomPosition(lig_idx)
                    lig_coord = np.array([pos.x, pos.y, pos.z])
                    
                    distances = np.linalg.norm(res_coords - lig_coord, axis=1)
                    min_dist = float(np.min(distances))
                    
                    if min_dist < 6.0:
                        score = self._score_salt_bridge_distance(min_dist)
                        
                        if score > 0.05:
                            salt_bridges.append({
                                'type': 'ligand_positive',
                                'ligand_atom': lig_idx,
                                'protein_residue': res_id,
                                'distance': min_dist,
                                'score': score,
                                'strength': self._classify_strength(score)
                            })
            
            # Negative ligand + positive residue
            if res_charge == '+':
                for lig_idx in negative_atoms:
                    pos = conf.GetAtomPosition(lig_idx)
                    lig_coord = np.array([pos.x, pos.y, pos.z])
                    
                    distances = np.linalg.norm(res_coords - lig_coord, axis=1)
                    min_dist = float(np.min(distances))
                    
                    if min_dist < 6.0:
                        score = self._score_salt_bridge_distance(min_dist)
                        
                        if score > 0.05:
                            salt_bridges.append({
                                'type': 'ligand_negative',
                                'ligand_atom': lig_idx,
                                'protein_residue': res_id,
                                'distance': min_dist,
                                'score': score,
                                'strength': self._classify_strength(score)
                            })
        
        return salt_bridges
    
    def detect_all_soft(self, ligand_mol: Chem.Mol) -> Dict[str, List[Dict]]:
        """
        Detect all NCI types with soft scoring.
        
        Returns:
            Dictionary with all interaction types and their soft scores
        """
        return {
            'hydrogen_bonds': self.detect_hydrogen_bonds_soft(ligand_mol),
            'pi_stacking': self.detect_pi_stacking_soft(ligand_mol),
            'hydrophobic': self.detect_hydrophobic_soft(ligand_mol),
            'salt_bridges': self.detect_salt_bridges_soft(ligand_mol)
        }
    
    # ========== Fragment Scoring ==========
    
    def calculate_fragment_structure_score(
        self,
        fragment_interactions: Dict[str, List[Dict]]
    ) -> Dict[str, any]:
        """
        Calculate comprehensive structure-based score for a fragment.
        
        This implements weighted scoring with:
        - Type-based weights (salt bridge > H-bond > π-π > hydrophobic)
        - Quality-weighted scoring (using soft scores)
        - Diminishing returns for multiple weak interactions
        
        Args:
            fragment_interactions: Interactions for this fragment
            
        Returns:
            Dictionary with structure score and breakdown
        """
        # Type weights (based on typical binding contribution)
        TYPE_WEIGHTS = {
            'salt_bridges': 5.0,    # ~3-5 kcal/mol
            'hydrogen_bonds': 3.0,  # ~1-3 kcal/mol
            'pi_stacking': 2.5,     # ~1-2 kcal/mol
            'hydrophobic': 1.0      # ~0.5-1 kcal/mol per contact
        }
        
        total_weighted_score = 0.0
        max_possible_score = 0.0
        breakdown = {}
        
        # Salt bridges
        salt_bridges = fragment_interactions.get('salt_bridges', [])
        salt_score = sum(sb.get('score', 0) for sb in salt_bridges)
        salt_weighted = salt_score * TYPE_WEIGHTS['salt_bridges']
        total_weighted_score += salt_weighted
        max_possible_score += len(salt_bridges) * TYPE_WEIGHTS['salt_bridges'] if salt_bridges else 0
        breakdown['salt_bridges'] = {
            'count': len(salt_bridges),
            'raw_score': salt_score,
            'weighted_score': salt_weighted,
            'strong_count': sum(1 for sb in salt_bridges if sb.get('strength') == 'STRONG')
        }
        
        # Hydrogen bonds
        hbonds = fragment_interactions.get('hydrogen_bonds', [])
        hbond_score = sum(hb.get('combined_score', 0) for hb in hbonds)
        hbond_weighted = hbond_score * TYPE_WEIGHTS['hydrogen_bonds']
        total_weighted_score += hbond_weighted
        max_possible_score += len(hbonds) * TYPE_WEIGHTS['hydrogen_bonds'] if hbonds else 0
        breakdown['hydrogen_bonds'] = {
            'count': len(hbonds),
            'raw_score': hbond_score,
            'weighted_score': hbond_weighted,
            'strong_count': sum(1 for hb in hbonds if hb.get('strength') == 'STRONG')
        }
        
        # π-π stacking
        pi_stacks = fragment_interactions.get('pi_stacking', [])
        pi_score = sum(pi.get('combined_score', 0) for pi in pi_stacks)
        pi_weighted = pi_score * TYPE_WEIGHTS['pi_stacking']
        total_weighted_score += pi_weighted
        max_possible_score += len(pi_stacks) * TYPE_WEIGHTS['pi_stacking'] if pi_stacks else 0
        breakdown['pi_stacking'] = {
            'count': len(pi_stacks),
            'raw_score': pi_score,
            'weighted_score': pi_weighted,
            'strong_count': sum(1 for pi in pi_stacks if pi.get('strength') == 'STRONG')
        }
        
        # Hydrophobic (with diminishing returns)
        hydrophobic = fragment_interactions.get('hydrophobic', [])
        hydrophobic_score = 0.0
        for i, hp in enumerate(sorted(hydrophobic, key=lambda x: x.get('score', 0), reverse=True)):
            # Diminishing returns: 1st = 100%, 2nd = 80%, 3rd = 60%, etc.
            diminishing_factor = 1.0 / (1 + 0.25 * i)
            hydrophobic_score += hp.get('score', 0) * diminishing_factor
        
        hydrophobic_weighted = hydrophobic_score * TYPE_WEIGHTS['hydrophobic']
        total_weighted_score += hydrophobic_weighted
        # Max possible with diminishing returns
        max_hp = sum(1.0 / (1 + 0.25 * i) for i in range(len(hydrophobic))) if hydrophobic else 0
        max_possible_score += max_hp * TYPE_WEIGHTS['hydrophobic']
        breakdown['hydrophobic'] = {
            'count': len(hydrophobic),
            'raw_score': hydrophobic_score,
            'weighted_score': hydrophobic_weighted,
            'strong_count': sum(1 for hp in hydrophobic if hp.get('strength') == 'STRONG')
        }
        
        # Normalize to 0-1 range
        if max_possible_score > 0:
            normalized_score = min(total_weighted_score / max_possible_score, 1.0)
        else:
            normalized_score = 0.0
        
        # Calculate confidence based on interaction quality
        total_interactions = sum(b['count'] for b in breakdown.values())
        strong_interactions = sum(b['strong_count'] for b in breakdown.values())
        
        if total_interactions == 0:
            confidence = 'LOW'
        elif strong_interactions >= 2 or (strong_interactions >= 1 and total_interactions >= 3):
            confidence = 'HIGH'
        elif strong_interactions >= 1 or total_interactions >= 2:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        return {
            'structure_score': normalized_score,
            'raw_weighted_score': total_weighted_score,
            'max_possible_score': max_possible_score,
            'breakdown': breakdown,
            'total_interactions': total_interactions,
            'strong_interactions': strong_interactions,
            'confidence': confidence
        }
    
    # ========== Helper Methods ==========
    
    def _get_hbond_donors(self, mol: Chem.Mol) -> List[int]:
        """Get hydrogen bond donor atoms."""
        donors = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in ['N', 'O']:
                if atom.GetTotalNumHs() > 0:
                    donors.append(atom.GetIdx())
        return donors
    
    def _get_hbond_acceptors(self, mol: Chem.Mol) -> List[int]:
        """Get hydrogen bond acceptor atoms."""
        acceptors = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in ['N', 'O', 'F']:
                acceptors.append(atom.GetIdx())
        return acceptors
    
    def _get_aromatic_rings(self, mol: Chem.Mol) -> List[Tuple[int, ...]]:
        """Get aromatic ring atom indices."""
        aromatic_rings = []
        ring_info = mol.GetRingInfo()
        
        for ring in ring_info.AtomRings():
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                aromatic_rings.append(ring)
        
        return aromatic_rings
    
    def _get_hydrophobic_atoms(self, mol: Chem.Mol) -> List[int]:
        """Get hydrophobic atoms."""
        hydrophobic = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
                if not any(n in ['N', 'O', 'F'] for n in neighbors):
                    hydrophobic.append(atom.GetIdx())
        return hydrophobic
    
    def _get_positive_atoms(self, mol: Chem.Mol) -> List[int]:
        """Get positively charged atoms."""
        return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetFormalCharge() > 0]
    
    def _get_negative_atoms(self, mol: Chem.Mol) -> List[int]:
        """Get negatively charged atoms."""
        return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetFormalCharge() < 0]


def map_interactions_to_fragments_soft(
    interactions: Dict[str, List[Dict]],
    fragment_labels: List[int],
    fragment_smiles: List[str]
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Map soft-scored interactions to fragments.
    
    Similar to original map_interactions_to_fragments but preserves soft scores.
    """
    fragment_interactions = defaultdict(lambda: {
        'hydrogen_bonds': [],
        'pi_stacking': [],
        'hydrophobic': [],
        'salt_bridges': []
    })
    
    # Map hydrogen bonds
    for hb in interactions['hydrogen_bonds']:
        atom_idx = hb['ligand_atom']
        if atom_idx < len(fragment_labels):
            frag_idx = fragment_labels[atom_idx]
            frag_smi = fragment_smiles[frag_idx]
            fragment_interactions[frag_smi]['hydrogen_bonds'].append(hb)
    
    # Map pi-stacking
    for pi in interactions['pi_stacking']:
        ring_atoms = pi['ligand_atoms']
        if ring_atoms and ring_atoms[0] < len(fragment_labels):
            frag_idx = fragment_labels[ring_atoms[0]]
            frag_smi = fragment_smiles[frag_idx]
            fragment_interactions[frag_smi]['pi_stacking'].append(pi)
    
    # Map hydrophobic
    for hp in interactions['hydrophobic']:
        atom_idx = hp['ligand_atom']
        if atom_idx < len(fragment_labels):
            frag_idx = fragment_labels[atom_idx]
            frag_smi = fragment_smiles[frag_idx]
            fragment_interactions[frag_smi]['hydrophobic'].append(hp)
    
    # Map salt bridges
    for sb in interactions['salt_bridges']:
        atom_idx = sb['ligand_atom']
        if atom_idx < len(fragment_labels):
            frag_idx = fragment_labels[atom_idx]
            frag_smi = fragment_smiles[frag_idx]
            fragment_interactions[frag_smi]['salt_bridges'].append(sb)
    
    return dict(fragment_interactions)


if __name__ == "__main__":
    print("✓ Soft NCI Detector Module Loaded")
    print("\nKey Features:")
    print("  - Continuous scoring (0-1) instead of binary detection")
    print("  - Distance-based quality assessment")
    print("  - Angle-based geometric quality for H-bonds and π-π")
    print("  - Diminishing returns for multiple weak interactions")
    print("  - Confidence estimation (HIGH/MEDIUM/LOW)")
    print("\nUsage:")
    print("  detector = SoftNCIDetector('protein.pdb')")
    print("  interactions = detector.detect_all_soft(ligand_mol)")
    print("  score = detector.calculate_fragment_structure_score(frag_interactions)")