import RNA

def prob(seq, ss, scale=True):
    """viennaRNA boltzmann probability"""
    fc = RNA.fold_compound(seq)
    if scale:
        _, mfe = fc.mfe()
        fc.exp_params_rescale(mfe)
    fc.pf()
    pr = fc.pr_structure(ss)
    return pr

def ensemble_defect(seq, ss, scale=True):
    fc = RNA.fold_compound(seq)
    if scale:
        _, mfe = fc.mfe()
        fc.exp_params_rescale(mfe)
    fc.pf()
    fc.bpp()
    ed = fc.ensemble_defect(ss)
    return ed

def print_subopt_result(structure, energy, data):
    ss_list = []
    if not structure == None:
        data['ss_list'].append((energy, structure))
        data['counter'] = data['counter'] + 1

def subopt(seq, e=0):
    fc = RNA.fold_compound(seq)
    fc.subopt_cb(e, print_subopt_result, subopt_data)
    subopt_data['ss_list'] = sorted(subopt_data['ss_list'])
    return subopt_data

def mfe(seq):
    fc = RNA.fold_compound(seq)
    ss = fc.mfe()
    return ss

def structural_seq_dist(seq, ss):
    ss_mfe = mfe(seq)[0]
    stk = []
    mp = {}

    for j, c in enumerate(ss):
        if c == '(':
            stk.append(j)
        elif c == ')':
            i = stk.pop()
            mp[j] = i
            mp[i] = j
        else:
            mp[j] = -1

    dist = len(ss)
    for j, c in enumerate(ss_mfe):
        if c == '(':
            stk.append(j)
        elif c == ')':
            i = stk.pop()
            
            if mp[j] == i:
                dist -= 2
        else:
            if mp[j] == -1:
                dist -= 1

    return dist

def structural_dist(ss_mfe, ss):
    stk = []
    mp = {}

    for j, c in enumerate(ss):
        if c == '(':
            stk.append(j)
        elif c == ')':
            i = stk.pop()
            mp[j] = i
            mp[i] = j
        else:
            mp[j] = -1

    dist = len(ss)
    for j, c in enumerate(ss_mfe):
        if c == '(':
            stk.append(j)
        elif c == ')':
            i = stk.pop()
            
            if mp[j] == i:
                dist -= 2
        else:
            if mp[j] == -1:
                dist -= 1

    return dist

def energy(seq, ss):
    fc = RNA.fold_compound(seq)
    return fc.eval_structure(ss)

def e_diff(seq, ss):
    ss_mfe = mfe(seq)[0]
    return abs(energy(seq, ss_mfe) - energy(seq, ss))

def eval_design(seq, ss, scale=True):
    subopt_data = { 'counter' : 0, 'sequence' : seq, 'ss_list': []}
    
    fc = RNA.fold_compound(seq)
    if scale:
        _, mfe = fc.mfe() # Computes the MFE structure
        fc.exp_params_rescale(mfe)
    fc.pf()  # Compute the partition function for a given RNA sequence
    fc.bpp() # Compute the base pair probability matrix for a given RNA sequence
    fc.subopt_cb(0, print_subopt_result, subopt_data) # Find all MFE structures

    pr = fc.pr_structure(ss)    # Compute the equilibrium probability of a particular secondary structure
    ed = fc.ensemble_defect(ss) # Compute the Ensemble Defect for a given target structure
    
    mfe_structs = list(sorted([st for e, st in subopt_data['ss_list']], key=lambda x: structural_dist(x, ss)))
    is_mfe = ss in mfe_structs
    is_umfe = is_mfe and subopt_data['counter'] == 1

    dist = structural_dist(mfe_structs[0], ss)
    energy_diff = e_diff(seq, ss)

    results = {
        "sequence": seq,
        "target_structure": ss,
        "mfe_structures": mfe_structs,
        "structural_dist": dist,
        "probability": float(f"{pr:.6f}"), #min(2.0, float(f"{pr:.6f}")),
        "ensemble_defect": float(f"{ed:.6f}"),
        "energy_diff": float(f"{energy_diff:.4f}"),
        "is_mfe": is_mfe,
        "is_umfe": is_umfe,
    }

    return results

if __name__ == "__main__":
    import json
    def pretty_print_dict(d):
        print(json.dumps(d, indent=4))
    seq = "GGCCCGAAAAAGGGCC"
    ss = "(((((......)))))"
    #seq = "GGCGGCGCCCCCGGCGUUUUCCGGAGGAGGGCCGCC"
    #seq = "GCCGGCGCCCCCGGCGAAGGCCGGAGGAGGGCCGGC"
    #ss = "((((((.((((((((....))))).)).).))))))"
    #seq = "AGGGGGAAAAAAAAGGGGAAAACCCCAACCCCCAAAAAAA"
    #ss  = ".(((((........)((((....))))..))))......."
    pretty_print_dict(eval_design(seq, ss))