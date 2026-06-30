# Vendored dependencies for merge-noise-phasemap.
#
# These three modules were copied verbatim (algorithm logic byte-for-byte) from the
# sibling experiment ../mechanism-recombination/ so this experiment is self-contained
# alongside the shipped decode_core/:
#   - domain_lattice.py : integer-sum Viterbi lattice substrate (constants + instances)
#   - gen_fair.py       : competence-p fair per-step generator P_p(move|state)
#   - arms_ext.py       : best_of_k_isobudget (selection baseline) + beam_no_merge (Phi-ablation)
#
# Only the decode_core sys.path resolution in arms_ext.py was adjusted for the new
# location; everything else is unchanged.
