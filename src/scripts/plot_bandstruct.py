"""
Plot band structures from HSE, normal GFN1, and modified GFN1 calculations.

HSE06 results are in gfn1_optim/exp/hse/vasprun.xml
DFTB results for first GFN are in gfn1_optim/exp/gfnpure/
DFTB results for fixed GFN are in gfn1_optim/exp/fixed/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_dftbp_band_out(band_out_file):
    """
    Parse eigenvalues and occupations from DFTB+ band.out file.
    Organizes by k-point for band structure plotting.
    
    Returns:
        dict: {kpoint_index: {'eigenvalues': array, 'occupations': array}}
    """
    assert Path(band_out_file).exists(), f"File not found: {band_out_file}"
    
    kpoints_data = {}
    current_kpt = None
    eigenvalues = []
    occupations = []
    
    with open(band_out_file, 'r') as f:
        for line in f:
            if 'KPT' in line and 'SPIN' in line:
                # Save previous k-point data if exists
                if current_kpt is not None and eigenvalues:
                    kpoints_data[current_kpt] = {
                        'eigenvalues': np.array(eigenvalues),
                        'occupations': np.array(occupations)
                    }
                
                # Extract k-point number
                parts = line.split()
                kpt_idx = None
                for i, part in enumerate(parts):
                    if part == 'KPT' and i + 1 < len(parts):
                        try:
                            kpt_idx = int(parts[i + 1])
                            break
                        except ValueError:
                            pass
                
                if kpt_idx is not None:
                    current_kpt = kpt_idx
                    eigenvalues = []
                    occupations = []
                continue
            
            # Parse eigenvalue line: band_idx energy occupation
            parts = line.split()
            if len(parts) >= 3:
                try:
                    energy_ev = float(parts[1])
                    occupation = float(parts[2])
                    eigenvalues.append(energy_ev)
                    occupations.append(occupation)
                except (ValueError, IndexError):
                    continue
    
    # Save last k-point
    if current_kpt is not None and eigenvalues:
        kpoints_data[current_kpt] = {
            'eigenvalues': np.array(eigenvalues),
            'occupations': np.array(occupations)
        }
    
    return kpoints_data if kpoints_data else None


def parse_vasp_vasprun(vasprun_file):
    """
    Parse eigenvalues and occupations from VASP vasprun.xml file.
    Organizes by k-point for band structure plotting.
    
    Returns:
        dict: {kpoint_index: {'eigenvalues': array, 'occupations': array}}
    """
    assert Path(vasprun_file).exists(), f"File not found: {vasprun_file}"
    
    tree = ET.parse(vasprun_file)
    root = tree.getroot()
    
    kpoints_data = {}
    
    # Find eigenvalues section
    eigenvalues_elem = root.find('.//eigenvalues')
    if eigenvalues_elem is None:
        return None
    
    # Get first spin channel (spin 1)
    spin_elem = None
    for set_elem in eigenvalues_elem.findall('.//set'):
        if set_elem.get('comment') == 'spin 1':
            spin_elem = set_elem
            break
    
    # If not found with comment, try finding first set within array
    if spin_elem is None:
        array_elem = eigenvalues_elem.find('array')
        if array_elem is not None:
            first_set = array_elem.find('set')
            if first_set is not None:
                spin_elem = first_set.find('set')
    
    if spin_elem is None:
        return None
    
    # Iterate over all k-points (direct children sets of spin_elem)
    for kpoint_elem in spin_elem.findall('set'):
        # Extract k-point index from comment
        comment = kpoint_elem.get('comment', '')
        kpt_idx = None
        if 'kpoint' in comment.lower():
            try:
                kpt_idx = int(comment.split()[-1])
            except (ValueError, IndexError):
                pass
        
        # If no comment or failed to parse, use sequential numbering
        if kpt_idx is None:
            kpt_idx = len(kpoints_data) + 1
        
        eigenvalues = []
        occupations = []
        
        for r_elem in kpoint_elem.findall('r'):
            # Format: <r>  energy  occupation </r>
            values = r_elem.text.split()
            if len(values) >= 2:
                try:
                    energy_ev = float(values[0])
                    occupation = float(values[1])
                    eigenvalues.append(energy_ev)
                    occupations.append(occupation)
                except (ValueError, IndexError):
                    continue
        
        if eigenvalues:
            kpoints_data[kpt_idx] = {
                'eigenvalues': np.array(eigenvalues),
                'occupations': np.array(occupations)
            }
    
    return kpoints_data if kpoints_data else None


def plot_band_structures(hse_vasprun, gfnpure_band_out, fixed_band_out, output_file=None):
    """
    Plot band structures from HSE, normal GFN1, and modified GFN1 calculations.
    Each band is plotted as energy vs k-point index.
    """
    # Parse all three band structures
    print("Parsing HSE band structure...")
    hse_data = parse_vasp_vasprun(hse_vasprun)
    assert hse_data is not None, "Failed to parse HSE vasprun.xml"
    print(f"  Found {len(hse_data)} k-point(s)")
    
    print("Parsing normal GFN1 band structure...")
    gfnpure_data = parse_dftbp_band_out(gfnpure_band_out)
    assert gfnpure_data is not None, "Failed to parse normal GFN1 band.out"
    print(f"  Found {len(gfnpure_data)} k-point(s)")
    
    print("Parsing modified GFN1 band structure...")
    fixed_data = parse_dftbp_band_out(fixed_band_out)
    assert fixed_data is not None, "Failed to parse modified GFN1 band.out"
    print(f"  Found {len(fixed_data)} k-point(s)")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get k-point indices sorted
    hse_kpts = sorted(hse_data.keys())
    gfnpure_kpts = sorted(gfnpure_data.keys())
    fixed_kpts = sorted(fixed_data.keys())
    
    # Determine maximum number of bands across all calculations
    max_bands = 0
    for kpt_data in [hse_data, gfnpure_data, fixed_data]:
        for kpt in kpt_data.values():
            max_bands = max(max_bands, len(kpt['eigenvalues']))
    
    # Plot each band across k-points
    # For each band index, plot its energy at each k-point
    
    # HSE data
    if hse_kpts:
        for band_idx in range(max_bands):
            band_energies = []
            band_kpts_x = []
            for kpt_idx in hse_kpts:
                eig = hse_data[kpt_idx]['eigenvalues']
                if band_idx < len(eig):
                    band_energies.append(eig[band_idx])
                    band_kpts_x.append(kpt_idx)
            
            if band_energies:
                ax.plot(band_kpts_x, band_energies, 'o-', color='red', 
                       markersize=2, linewidth=0.8, alpha=0.6, label='HSE06' if band_idx == 0 else None)
    
    # GFN1 normal data
    if gfnpure_kpts:
        for band_idx in range(max_bands):
            band_energies = []
            band_kpts_x = []
            for kpt_idx in gfnpure_kpts:
                eig = gfnpure_data[kpt_idx]['eigenvalues']
                if band_idx < len(eig):
                    band_energies.append(eig[band_idx])
                    band_kpts_x.append(kpt_idx)
            
            if band_energies:
                ax.plot(band_kpts_x, band_energies, 's-', color='blue', 
                       markersize=2, linewidth=0.8, alpha=0.6, label='GFN1 (normal)' if band_idx == 0 else None)
    
    # GFN1 modified data
    if fixed_kpts:
        for band_idx in range(max_bands):
            band_energies = []
            band_kpts_x = []
            for kpt_idx in fixed_kpts:
                eig = fixed_data[kpt_idx]['eigenvalues']
                if band_idx < len(eig):
                    band_energies.append(eig[band_idx])
                    band_kpts_x.append(kpt_idx)
            
            if band_energies:
                ax.plot(band_kpts_x, band_energies, '^-', color='green', 
                       markersize=2, linewidth=0.8, alpha=0.6, label='GFN1 (modified)' if band_idx == 0 else None)
    
    # Add labels and title
    ax.set_xlabel('K-point Index', fontsize=12)
    ax.set_ylabel('Energy (eV)', fontsize=12)
    ax.set_title('Band Structure Comparison', fontsize=14, fontweight='bold')
    
    # Only show legend if we have multiple methods
    if len(hse_kpts) > 0 or len(gfnpure_kpts) > 0 or len(fixed_kpts) > 0:
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    if output_file is None:
        output_file = Path(__file__).parent.parent.parent / 'results' / 'bandstructure_comparison.png'
    else:
        output_file = Path(output_file)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nBand structure plot saved: {output_file}")
    
    plt.close(fig)


if __name__ == '__main__':
    # Paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    hse_vasprun = project_root / 'exp' / 'hse' / 'vasprun.xml'
    gfnpure_band_out = project_root / 'exp' / 'gfnpure' / 'band.out'
    fixed_band_out = project_root / 'exp' / 'fixed' / 'band.out'
    
    plot_band_structures(hse_vasprun, gfnpure_band_out, fixed_band_out)
