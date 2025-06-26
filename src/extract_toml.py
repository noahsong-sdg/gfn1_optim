import re

def create_element_toml(element_list, output_filename):
    """
    Parse element data from defaults_raw.dat and write to TOML file.
    
    Args:
        element_list: List of element symbols (e.g., ['H', 'C', 'N'])
        output_filename: Name of the output TOML file
    """
    # Convert to set for O(1) lookup
    target_elements = set(element_list)
    
    # Read the raw data file
    with open('defaults_raw.dat', 'r') as f:
        lines = f.readlines()
    
    # Dictionary to store parsed element data
    elements_data = {}
    current_element = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        
        # Check if line starts with an element (starts with capital letter, not a float)
        if parts[0].isalpha() and parts[0][0].isupper():
            # New element found
            current_element = parts[0]
            
            # Only process if this element is in our target list
            if current_element in target_elements:
                # Parse the 4 scalar values
                gam = float(parts[1])
                gam3 = float(parts[2])
                arep = float(parts[3])
                zeff = float(parts[4])
                shell = str(parts[5])  # orbital name (e.g., "1s")
                shpoly_val = float(parts[6])
                lgam_val = float(parts[7])
                level_val = float(parts[8])
                slater_val = float(parts[9])
                
                # Initialize the element data
                elements_data[current_element] = {
                    'gam': gam,
                    'gam3': gam3,
                    'arep': arep,
                    'zeff': zeff,
                    'shells': [],
                    'shpoly': [],
                    'lgam': [],
                    'levels': [],
                    'slater': []
                }
                    
                elements_data[current_element]['shells'].append(shell)
                elements_data[current_element]['shpoly'].append(shpoly_val)
                elements_data[current_element]['lgam'].append(lgam_val)
                elements_data[current_element]['levels'].append(level_val)
                elements_data[current_element]['slater'].append(slater_val)
            else:
                # Reset current_element if we're not interested in this element
                current_element = None
        
        elif current_element and len(parts) >= 5:
            # Continuation line with orbital data (we already know current_element is in target_elements)
            shell = parts[0]  # orbital name (e.g., "2s", "2p")
            shpoly_val = float(parts[1])
            lgam_val = float(parts[2])
            level_val = float(parts[3])
            slater_val = float(parts[4])
            
            elements_data[current_element]['shells'].append(shell)
            elements_data[current_element]['shpoly'].append(shpoly_val)
            elements_data[current_element]['lgam'].append(lgam_val)
            elements_data[current_element]['levels'].append(level_val)
            elements_data[current_element]['slater'].append(slater_val)
        
        # Early termination: if we've found all requested elements, we can stop
        if len(elements_data) == len(target_elements):
            break
    
    # Write to TOML file
    with open(output_filename, 'w') as f:
        for element in element_list:
            if element in elements_data:
                data = elements_data[element]
                f.write(f"[element.{element}]\n")
                f.write(f"gam = {data['gam']} # atomic hubbard parameter\n")
                f.write(f"gam3 = {data['gam3']} # atomic hubbard derivative\n")
                f.write(f"arep = {data['arep']} # alpha_A\n")
                f.write(f"zeff = {data['zeff']} # effective nuclear charge\n")
                
                # Format arrays
                shells_str = ' '.join([f'"{shell}"' for shell in data['shells']])
                f.write(f"shells = [ {shells_str} ]\n")
                
                shpoly_str = ' '.join([str(val) for val in data['shpoly']])
                f.write(f"shpoly = [ {shpoly_str} ] # polynomial enhancement factor\n")
                
                lgam_str = ' '.join([str(val) for val in data['lgam']])
                f.write(f"lgam = [ {lgam_str} ]\n")
                
                levels_str = ' '.join([str(val) for val in data['levels']])
                f.write(f"levels = [ {levels_str} ] # atomic level energies\n")
                
                slater_str = ' '.join([str(val) for val in data['slater']])
                f.write(f"slater = [ {slater_str} ] # Slater exponents\n")
                
                f.write("\n")  # Empty line between elements
            else:
                print(f"Warning: Element {element} not found in data file")

# Example usage:
# create_element_toml(['H', 'C', 'N'], 'selected_elements.toml')
