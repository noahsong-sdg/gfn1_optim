me of the output file
OUTPUT_CSV="results.csv"

# Write the header row to the CSV file.
echo "Structure,Bandgap(eV),FreeEnergy(eV)" > ${OUTPUT_CSV}

# This awk program now calculates the gap AND grabs the free energy.
AWK_SCRIPT='
# Rule to find the final free energy (TOTEN)
# This looks for a line where the first three fields match exactly.
$1=="free" && $2=="energy" && $3=="TOTEN" {
  toten = $5 # The energy value is the 5th field
  }

  # Rule to find the VBM and CBM for the bandgap
  /E-fermi/ {
    vbm=""; cbm=""
    }
    NF == 3 && $1 ~ /^[0-9]+$/ {
      if ($3 > 0) {
	          vbm = $2
		    }
		      if ($3 == 0 && vbm != "" && cbm == "") {
			          cbm = $2
				    }
			    }

			    # At the end of the file, print both results, separated by a comma.
			    END {
			      if (vbm != "" && cbm != "" && toten != "") {
				          printf "%.4f,%.8f", cbm - vbm, toten
					    }
				    }
				    '

				    # Loop through all directories starting with "structure_"
				    for dir in structure_*; do
					      if [ -d "$dir" ]; then
						          if [ -r "$dir/OUTCAR" ];
								      then
									            # Run the awk script and capture the comma-separated output
										          results=$(awk "$AWK_SCRIPT" "$dir/OUTCAR")
											        
											        # If the 'results' variable is not empty, write to the CSV
												      if [ -n "$results" ]; then
													              echo "${dir},${results}" >> ${OUTPUT_CSV}
														            else
																            echo "${dir},Data_Not_Found,Data_Not_Found" >> ${OUTPUT_CSV}
																	          fi
																		        
																		      else
																			            echo "${dir},OUTCAR_Missing,OUTCAR_Missing" >> ${OUTPUT_CSV}
																				        fi
																					  fi
																				  done

																				  echo "Extraction complete. 👍 Results are in ${OUTPUT_CSV}"
