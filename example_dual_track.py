"""
Example: Dual-Track Fragment Analysis for Diffusion Model Conditioning

This example demonstrates the complete dual-track analysis workflow:

Track 1 (Structure): Soft NCI Detection
  - Continuous scoring instead of hard thresholds
  - Distance/angle quality assessment
  - Interaction strength classification

Track 2 (Knowledge): Chemical Knowledge Analysis
  - LLM-based pharmacophore analysis
  - Pocket complementarity assessment
  - Drug-likeness evaluation

Score Fusion:
  - Confidence-weighted combination
  - Score agreement assessment
  - Final ranking with confidence levels

Output: High-quality fragment selection for diffusion model input

Usage:
  # With LLM (recommended)
  python example_dual_track.py --api-key YOUR_API_KEY
  
  # Without LLM (rule-based fallback)
  python example_dual_track.py --no-llm

Author: For diffusion model conditioning
"""

import os
import sys
import json
import argparse
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem

# Import dual-track modules
try:
    from dual_track_analyzer import DualTrackFragmentAnalyzer
    from pocket_features import extract_pocket_features_from_pdb
    DUAL_TRACK_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print("Please ensure all modules are in the same directory")
    DUAL_TRACK_AVAILABLE = False


def create_test_data():
    """
    Create test complex for demonstration.
    
    Creates a simple protein-ligand complex with:
    - A small protein fragment with various residue types
    - A drug-like ligand (benzoic acid derivative)
    """
    test_dir = tempfile.mkdtemp()
    
    print("📦 Creating test complex...")
    
    # Protein structure with diverse residues
    pdb_content = """HEADER    TEST PROTEIN FOR DUAL-TRACK ANALYSIS
ATOM      1  N   SER A   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  SER A   1      11.000  10.000  10.000  1.00 20.00           C
ATOM      3  C   SER A   1      11.500  11.000  10.000  1.00 20.00           C
ATOM      4  O   SER A   1      11.500  11.500  11.000  1.00 20.00           O
ATOM      5  CB  SER A   1      11.500   9.000  10.000  1.00 20.00           C
ATOM      6  OG  SER A   1      12.500   9.000  10.500  1.00 20.00           O
ATOM      7  N   PHE A   2      15.000  10.000  10.000  1.00 20.00           N
ATOM      8  CA  PHE A   2      16.000  10.000  10.000  1.00 20.00           C
ATOM      9  C   PHE A   2      16.500  11.000  10.000  1.00 20.00           C
ATOM     10  O   PHE A   2      16.500  11.500  11.000  1.00 20.00           O
ATOM     11  CB  PHE A   2      16.500   9.000  10.000  1.00 20.00           C
ATOM     12  CG  PHE A   2      17.500   9.000  10.500  1.00 20.00           C
ATOM     13  CD1 PHE A   2      18.000  10.000  11.000  1.00 20.00           C
ATOM     14  CD2 PHE A   2      18.000   8.000  10.500  1.00 20.00           C
ATOM     15  CE1 PHE A   2      19.000  10.000  11.500  1.00 20.00           C
ATOM     16  CE2 PHE A   2      19.000   8.000  11.000  1.00 20.00           C
ATOM     17  CZ  PHE A   2      19.500   9.000  11.500  1.00 20.00           C
ATOM     18  N   LEU A   3       8.000  12.000   8.000  1.00 20.00           N
ATOM     19  CA  LEU A   3       9.000  12.000   8.000  1.00 20.00           C
ATOM     20  C   LEU A   3       9.500  13.000   8.000  1.00 20.00           C
ATOM     21  O   LEU A   3       9.500  13.500   9.000  1.00 20.00           O
ATOM     22  CB  LEU A   3       9.500  11.000   8.000  1.00 20.00           C
ATOM     23  CG  LEU A   3      10.000  10.500   7.000  1.00 20.00           C
ATOM     24  CD1 LEU A   3      10.500   9.500   7.500  1.00 20.00           C
ATOM     25  CD2 LEU A   3      10.500  11.500   6.500  1.00 20.00           C
ATOM     26  N   ARG A   4      13.000  11.000  12.000  1.00 20.00           N
ATOM     27  CA  ARG A   4      14.000  11.000  12.000  1.00 20.00           C
ATOM     28  C   ARG A   4      14.500  12.000  12.000  1.00 20.00           C
ATOM     29  O   ARG A   4      14.500  12.500  13.000  1.00 20.00           O
ATOM     30  CB  ARG A   4      14.500  10.000  12.000  1.00 20.00           C
ATOM     31  CG  ARG A   4      15.000   9.500  11.000  1.00 20.00           C
ATOM     32  CD  ARG A   4      15.500   8.500  11.500  1.00 20.00           C
ATOM     33  NE  ARG A   4      16.000   8.000  10.500  1.00 20.00           N
ATOM     34  CZ  ARG A   4      16.500   7.000  10.500  1.00 20.00           C
ATOM     35  NH1 ARG A   4      17.000   6.500   9.500  1.00 20.00           N
ATOM     36  NH2 ARG A   4      16.500   6.500  11.500  1.00 20.00           N
ATOM     37  N   TYR A   5      11.000  13.000   9.000  1.00 20.00           N
ATOM     38  CA  TYR A   5      12.000  13.000   9.000  1.00 20.00           C
ATOM     39  C   TYR A   5      12.500  14.000   9.000  1.00 20.00           C
ATOM     40  O   TYR A   5      12.500  14.500  10.000  1.00 20.00           O
ATOM     41  CB  TYR A   5      12.500  12.000   9.000  1.00 20.00           C
ATOM     42  CG  TYR A   5      13.500  12.000   9.500  1.00 20.00           C
ATOM     43  CD1 TYR A   5      14.000  13.000  10.000  1.00 20.00           C
ATOM     44  CD2 TYR A   5      14.000  11.000   9.500  1.00 20.00           C
ATOM     45  CE1 TYR A   5      15.000  13.000  10.500  1.00 20.00           C
ATOM     46  CE2 TYR A   5      15.000  11.000  10.000  1.00 20.00           C
ATOM     47  CZ  TYR A   5      15.500  12.000  10.500  1.00 20.00           C
ATOM     48  OH  TYR A   5      16.500  12.000  11.000  1.00 20.00           O
END
"""
    
    pdb_file = os.path.join(test_dir, "test_protein.pdb")
    with open(pdb_file, 'w') as f:
        f.write(pdb_content)
    
    # Create a drug-like ligand (benzoic acid with additional groups)
    # This will decompose into multiple fragments
    ligand_smiles = "c1ccccc1C(=O)O"  # Benzoic acid
    
    mol = Chem.MolFromSmiles(ligand_smiles)
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    
    # Position in binding site
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (pos.x + 13.0, pos.y + 10.0, pos.z + 10.0))
    
    sdf_file = os.path.join(test_dir, "test_ligand.sdf")
    writer = Chem.SDWriter(sdf_file)
    writer.write(mol)
    writer.close()
    
    print(f"   ✓ Created test protein: {pdb_file}")
    print(f"   ✓ Created test ligand: {sdf_file}")
    print(f"   ✓ Ligand SMILES: {ligand_smiles}")
    
    return pdb_file, sdf_file


def run_dual_track_analysis(
    protein_pdb: str,
    ligand_sdf: str,
    api_key: str = None,
    use_llm: bool = True,
    output_dir: str = 'diffusion_input'
):
    """
    Run complete dual-track analysis.
    
    Args:
        protein_pdb: Path to protein PDB
        ligand_sdf: Path to ligand SDF
        api_key: DeepSeek API key (optional)
        use_llm: Whether to use LLM for Track 2
        output_dir: Output directory
    """
    print("\n" + "="*70)
    print("🚀 DUAL-TRACK FRAGMENT ANALYSIS")
    print("="*70)
    
    # Load ligand
    print("\n📂 Loading input files...")
    supplier = Chem.SDMolSupplier(ligand_sdf)
    ligand_mol = supplier[0]
    
    if ligand_mol is None:
        print("❌ Failed to load ligand")
        return None
    
    print(f"   ✓ Loaded ligand: {ligand_mol.GetNumAtoms()} atoms")
    
    # Extract pocket features
    print("\n🔬 Extracting pocket features...")
    pocket_features = extract_pocket_features_from_pdb(protein_pdb)
    print(f"   ✓ Found {pocket_features.get('num_residues', 0)} residues")
    
    # Initialize dual-track analyzer
    print("\n⚙️  Initializing Dual-Track Analyzer...")
    
    analyzer = DualTrackFragmentAnalyzer(
        api_key=api_key,
        fusion_strategy='confidence_weighted',
        use_llm_knowledge=use_llm
    )
    
    if use_llm and api_key:
        print("   Mode: LLM-enhanced (Track 2 uses DeepSeek)")
    else:
        print("   Mode: Rule-based (Track 2 uses chemical rules)")
    
    # Run analysis
    results = analyzer.analyze(
        ligand_mol=ligand_mol,
        protein_pdb=protein_pdb,
        pocket_features=pocket_features,
        save_coordinates=True,
        output_dir=output_dir
    )
    
    # Save complete results
    results_file = os.path.join(output_dir, 'analysis_results_dual_track.json')
    
    # Convert non-serializable items
    serializable_results = {
        'critical_fragments': results['critical_fragments'],
        'all_fragments': results['all_fragments'],
        'fragment_labels': results['fragment_labels'],
        'analysis_metadata': results['analysis_metadata'],
        'saved_files': results['saved_files']
    }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n📄 Saved analysis results: {results_file}")
    
    return results


def print_diffusion_guidance(output_dir: str):
    """Print guidance for using results with diffusion model."""
    print("\n" + "="*70)
    print("🎯 DIFFUSION MODEL INPUT GUIDANCE")
    print("="*70)
    
    print(f"""
📁 Output Files in {output_dir}/:

   Required Inputs for Diffusion Model:
   ────────────────────────────────────
   1. critical_fragment_rank1.sdf  ⭐ PRIMARY CONDITION
      → The most critical fragment to preserve/build around
      → Contains 3D coordinates aligned with binding site
   
   2. pocket.pdb
      → Protein binding pocket for spatial constraints
      → Use for pocket-conditioned generation
   
   Optional Inputs:
   ────────────────
   3. critical_fragment_rank2.sdf
      → Secondary important fragment
      
   4. critical_fragment_rank3.sdf  
      → Tertiary important fragment
      
   5. ligand_full.sdf
      → Original complete ligand (reference)
      
   6. coordinates.json
      → All fragment coordinates and metadata
      → Contains centroid positions for spatial conditioning

💡 Recommended Usage:

   # Basic conditioning (most common)
   diffusion_model.generate(
       condition_fragment='critical_fragment_rank1.sdf',
       pocket='pocket.pdb'
   )
   
   # Multi-fragment conditioning
   diffusion_model.generate(
       primary_fragment='critical_fragment_rank1.sdf',
       secondary_fragment='critical_fragment_rank2.sdf',
       pocket='pocket.pdb'
   )
   
   # Coordinate-based conditioning
   import json
   with open('coordinates.json') as f:
       coords = json.load(f)
   
   centroid = coords['critical_fragments'][0]['centroid']
   diffusion_model.generate(
       anchor_point=centroid,
       pocket='pocket.pdb'
   )

📊 Quality Indicators (from analysis):

   Check 'confidence' in analysis results:
   - HIGH: Strong candidate for conditioning
   - MEDIUM: Good candidate, verify results
   - LOW: Use with caution, may need alternatives

   Check 'score_agreement':
   - > 0.7: Both tracks agree → reliable
   - 0.4-0.7: Partial agreement → acceptable
   - < 0.4: Tracks disagree → verify manually
""")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Dual-Track Fragment Analysis for Diffusion Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With LLM (best quality)
  python example_dual_track.py --api-key YOUR_KEY

  # Without LLM (no API needed)
  python example_dual_track.py --no-llm
  
  # Custom input files
  python example_dual_track.py --protein protein.pdb --ligand ligand.sdf
        """
    )
    
    parser.add_argument('--api-key', type=str, default=None,
                        help='DeepSeek API key (or set DEEPSEEK_API_KEY env var)')
    parser.add_argument('--no-llm', action='store_true',
                        help='Use rule-based analysis instead of LLM')
    parser.add_argument('--protein', type=str, default=None,
                        help='Path to protein PDB file')
    parser.add_argument('--ligand', type=str, default=None,
                        help='Path to ligand SDF file')
    parser.add_argument('--output', type=str, default='diffusion_input',
                        help='Output directory (default: diffusion_input)')
    
    args = parser.parse_args()
    
    # Check module availability
    if not DUAL_TRACK_AVAILABLE:
        print("❌ Required modules not available")
        print("Please ensure these files are in the same directory:")
        print("  - dual_track_analyzer.py")
        print("  - soft_nci_detector.py")
        print("  - chemical_knowledge_analyzer.py")
        print("  - pocket_features.py")
        sys.exit(1)
    
    # Get API key
    api_key = args.api_key or os.getenv('DEEPSEEK_API_KEY')
    use_llm = not args.no_llm and api_key is not None
    
    if not args.no_llm and not api_key:
        print("⚠️  No API key provided, using rule-based analysis")
        print("   For LLM-enhanced analysis, provide --api-key or set DEEPSEEK_API_KEY")
        use_llm = False
    
    # Get input files
    if args.protein and args.ligand:
        protein_pdb = args.protein
        ligand_sdf = args.ligand
    else:
        print("\n📦 No input files provided, creating test data...")
        protein_pdb, ligand_sdf = create_test_data()
    
    # Run analysis
    results = run_dual_track_analysis(
        protein_pdb=protein_pdb,
        ligand_sdf=ligand_sdf,
        api_key=api_key,
        use_llm=use_llm,
        output_dir=args.output
    )
    
    if results:
        print_diffusion_guidance(args.output)
        print("\n✅ Analysis complete! Files ready for diffusion model.")
    else:
        print("\n❌ Analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()