#!/bin/bash
# Verification script for GaN-ZnO VASP input files
# Run this to check that all required files are present and configured correctly

echo "======================================================================"
echo "VASP INPUT FILES VERIFICATION FOR GaN-ZnO BAND STRUCTURE CALCULATIONS"
echo "======================================================================"
echo ""

DIRS="x0.00 x0.20 x0.40 x0.50 x0.60 x0.80 x1.00"
REQUIRED_FILES="POSCAR KPOINTS KPOINTS_bands INCAR INCAR_bands INCAR_pbe_relax POTCAR_instructions.txt"

# Check each directory
for DIR in $DIRS; do
    echo "Checking $DIR..."
    
    # Check directory exists
    if [ ! -d "$DIR" ]; then
        echo "  ERROR: Directory $DIR does not exist!"
        continue
    fi
    
    # Check required files
    MISSING=0
    for FILE in $REQUIRED_FILES; do
        if [ ! -f "$DIR/$FILE" ]; then
            echo "  ERROR: Missing $FILE"
            MISSING=1
        fi
    done
    
    if [ $MISSING -eq 0 ]; then
        echo "  OK: All required files present"
    fi
    
    # Verify MAGMOM matches POSCAR
    POSCAR_LINE=$(sed -n '7p' "$DIR/POSCAR")
    MAGMOM_LINE=$(grep "MAGMOM" "$DIR/INCAR")
    echo "  POSCAR elements: $POSCAR_LINE"
    echo "  INCAR MAGMOM: $MAGMOM_LINE"
    
    # Count atoms
    ATOM_COUNTS=$(echo $POSCAR_LINE | tr -d ' ')
    echo "  Atom counts: $ATOM_COUNTS"
    
    echo ""
done

echo "======================================================================"
echo "NEXT STEPS:"
echo "======================================================================"
echo ""
echo "1. Generate POTCAR files for each directory:"
echo "   - Read POTCAR_instructions.txt in each directory"
echo "   - Element order varies between compositions!"
echo ""
echo "2. Verify ENCUT vs POTCAR ENMAX:"
echo "   grep ENMAX POTCAR  (after generating)"
echo "   Should be <= 520 eV / 1.3 = 400 eV"
echo ""
echo "3. Test one composition before running all:"
echo "   Recommended: x0.00 or x1.00 (pure phases)"
echo ""
echo "4. For HSE06 calculations, expect:"
echo "   - 1-7 days per calculation"
echo "   - 2-4 GB memory per core"
echo "   - Significant disk space for WAVECAR/CHGCAR"
echo ""
echo "======================================================================"

