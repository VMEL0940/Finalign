# finalign_cli.py

import argparse
import os
import sys
import functions_global as FN
from datetime import datetime

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
import pandas as pd
import numpy as np

import math


from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score, pairwise_distances
from scipy.spatial.distance import pdist, squareform


def main():
    parser = argparse.ArgumentParser(description="FINALIGN: Resolve your MSA for analysis ")

    parser.add_argument("--input_fasta", required=True, help="Path to input MSA FASTA file")
    parser.add_argument("--metadata_csv", required=True, help="Path to metadata CSV file")
    parser.add_argument("--date", choices=["YMD", "Y"], default = "YMD", help = "If using full date: YMD or using only Year: Y")
    parser.add_argument("--out_dir", required=True, help="Directory to save output files")
    parser.add_argument("--out_name", required=True, help="Name of output files")
    parser.add_argument("--resolution_strategy", choices=["d", "p", "g"], default="p", help="Preferred resolution strategy")
    parser.add_argument("--resolution_mode", choices=["clean", "raw"], default="clean", help="Fallback behavior for single N")
    parser.add_argument("--proceed_trimming", choices=["yes", "no"], default="yes", help="Trim sequences to ORF")
    parser.add_argument("--n_threshold", type=float, default=0.02, help="Maximum allowed proportion of Ns (Recommend: 0.02)")
    parser.add_argument("--wgd", type=float, default=0.5, help="Weight of genetic distance (0–1 (Recommend: 0.5); Weight of time automatically 1-wgd)")

    args = parser.parse_args()


    # === Load and clean sequences ===
    cleaned_sequences_with_ids = FN.clean_sequences_with_ids(args.input_fasta, args.n_threshold)
    
    # === Trimming ORF ====
        
    if args.proceed_trimming == 'yes':
        print("==================== Trimming ORF ====================")
    
        # Assume raw sequences are in 'cleaned_sequences_with_ids'
        start_pos = FN.probable_start(cleaned_sequences_with_ids)
        stop_pos = FN.probable_stop(cleaned_sequences_with_ids)
        print(f"Probable Start Codon Position: {start_pos}")
        print(f"Probable Stop Codon Position: {stop_pos}")

        trimmed_seq = FN.cut_sequences(cleaned_sequences_with_ids)
        for seq_id, adjusted_seq in trimmed_seq:
            print(f"{seq_id}: {adjusted_seq[:15]} ~ {adjusted_seq[-15:]}")

        cleaned_trimmed_seq = FN.remove_gap_columns(trimmed_seq)

        # Optionally, export trimmed sequences to FASTA
        print(f"\n Trimming complete. {len(cleaned_trimmed_seq)} sequences prepared for downstream analysis.\n")

    else:
        cleaned_trimmed_seq = FN.remove_gap_columns(cleaned_sequences_with_ids)
        print("  Trimming skipped by user choice. Proceeding with full-length sequences.\n")
 
    alignment = MultipleSeqAlignment([SeqRecord(Seq(seq), id=seq_id) for seq_id, seq in cleaned_trimmed_seq])

    # === Load metadata and match to alignment ===
    df = pd.read_csv(args.metadata_csv)
    meta = df[df['name'].isin([r.id for r in alignment])].copy()
    meta = meta.set_index('name').loc[[r.id for r in alignment]].reset_index()
    dates_dict = dict(zip(meta['name'], meta['date']))
    # Choose your date format
    if args.date == "YMD" :
        date_format = "%Y.%M.%d"
    elif args.date == "Y" :
        date_format = "%Y"
    
    # === Global consensus ===
    global_consensus = FN.cons_matrix([seq for (_, seq) in cleaned_trimmed_seq])

    # === Distance clustering ===
    
    print("==================== Distance clustering ====================")
    dist_matrix, labels = FN.compute_distance_matrix(alignment)
    
    # Convert string dates to datetime objects
    try:
        parsed_dates = [datetime.strptime(str(dates_dict[label]), date_format) for label in labels]
    except ValueError as e:
        print("❌ Date parsing error. Check the format in your metadata.")
        raise e

    # Convert to number of days since the earliest date
    min_date = min(parsed_dates)
    numeric_dates = [(d - min_date).days for d in parsed_dates]
    
    # Determine optimal number of clusters using Silhouette Score
    silhouette_scores = []
    min_clusters = max(2, len(alignment) // 250)
    max_clusters = max(min_clusters + 1, len(alignment) // 50)
    range_n_clusters = range(math.floor(min_clusters), math.floor(max_clusters) + 1)

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(dist_matrix)
        silhouette_avg = silhouette_score(dist_matrix, cluster_labels, metric='precomputed')
        silhouette_scores.append(silhouette_avg)

    # Optimal number of clusters
    optimal_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
    
    wtd = 1-args.wgd
    
    dist_clusters = FN.subsample_and_assign_clusters(
        dist_matrix, labels, numeric_dates,
        n_clusters=optimal_n_clusters, sample_size=1, wgd=args.wgd, wtd=wtd
    )
        
    # mapping with sequences
    dist_cluster_groups = FN.sort_sequences_into_representative_clusters(cleaned_trimmed_seq, dist_clusters)
    
    # Distance-based clusters
    distance_membership = {}
    distance_consensus = {}

    for cluster_id, cluster_sequences in dist_cluster_groups.items():
        for seq_id, seq_str in cluster_sequences:
            distance_membership[seq_id] = cluster_id

        sequences_only = [seq_data[1] for seq_data in cluster_sequences]
        consensus_seq = FN.cons_matrix(sequences_only)
        print(f"  (📏 Distance-based Method) Cluster {cluster_id}: {len(cluster_sequences)} members")

        distance_consensus[cluster_id] = consensus_seq
    
    
    # === Phylogenetic clustering ===
    print("==================== Phylogenetic clustering ====================")
    
    converted = FN.convert_non_ACGT_to_gaps(cleaned_trimmed_seq)
    clusters_by_threshold, tmp_nwk_path, tmp_mapping = FN.build_fasttree_and_sweep_thresholds(
        converted, (0.03, 0.1 + 0.001, 0.001), 0.0, "max_clade", args.out_name, args.out_dir
    )
    elbow_threshold = FN.detect_elbow_threshold(clusters_by_threshold)
    phylo_clusters = clusters_by_threshold[elbow_threshold]
    
    phylo_cluster_groups = FN.sort_sequences_into_clusters(cleaned_trimmed_seq, phylo_clusters)
    
    # Phylogenetic-based clusters
    phylo_membership = {}
    phylo_consensus = {}

    for cluster_id, cluster_sequences in phylo_cluster_groups.items():
        for seq_id, seq_str in cluster_sequences:
            phylo_membership[seq_id] = cluster_id

        sequences_only = [seq_data[1] for seq_data in cluster_sequences]
        consensus_seq = FN.cons_matrix(sequences_only)
        print(f"  (🌲 Phylogenetic-based Method) Cluster {cluster_id}: {len(cluster_sequences)} members")

        phylo_consensus[cluster_id] = consensus_seq


    # === Step 1: IUPAC Ambiguity Resolution ===
    resolved_sequences = {}
    full_logs_amb = []            # Log for each ambiguous base (initial resolution)
    remaining_logs_amb = []       # Log for ambiguous positions that remain unresolved

    print("==================== Resolving ambiguous bases ====================")
    total_seqs = len(cleaned_trimmed_seq)
    
    # Process each sequence
    for idx, (seq_id, seq_str) in enumerate(cleaned_trimmed_seq):
        print(f"    [{idx}/{total_seqs}] Processing {seq_id}...", end='\r')
        full_log = [f"[{seq_id}]"]
        # Retrieve cluster IDs from the membership dictionaries.
        dist_cluster_id = distance_membership[seq_id]
        phylo_cluster_id = phylo_membership[seq_id]
        dist_con = distance_consensus[dist_cluster_id]
        phylo_con = phylo_consensus[phylo_cluster_id]

        min_len = min(len(seq_str), len(dist_con), len(phylo_con))
        seq_list = list(seq_str)
        ambiguous_positions = []

        full_log = [f"[{seq_id}]"]
        for i in range(min_len):
            actual_base = seq_str[i]
            if actual_base in FN.iupac_map:
                expansion_str = FN.show_expansion(actual_base, FN.iupac_map)
                d_base = dist_con[i]
                p_base = phylo_con[i]
                g_base = global_consensus[i]

                # Get corresponding codon index
                codon_index = i // 3
                start = codon_index * 3
                end = start + 3

                # Ensure codon range is within bounds
                if end > min_len:
                    continue

                codon_str = "".join(seq_list[start:end])

                # Resolve ambiguous base using updated function
                final_choice, resolved_aa, d_aa, p_aa, g_aa, type = FN.resolve_ambiguous(
                    actual_base, d_base, p_base, g_base, FN.iupac_map, FN.codon_table, codon_str, args.resolution_strategy
                )

                # Construct triplet representation (e.g., ATG (M))
                codon_triplet = f"{codon_str} ({FN.codon_table.get(codon_str, '-')})"
                d_triplet = f"{codon_str.replace(actual_base, d_base)} ({d_aa if d_aa else '-'})"
                p_triplet = f"{codon_str.replace(actual_base, p_base)} ({p_aa if p_aa else '-'})"
                g_triplet = f"{codon_str.replace(actual_base, g_base)} ({g_aa if g_aa else '-'})"

                # Full log entry with MSA codon information
                log_entry = (
                f"   Original codon {codon_str} (pos: {codon_index+1} ({start+1}-{end})) \n"
                f"   Dist_con={d_triplet}, Phylo_con={p_triplet}, Global_con={g_triplet}  \n"
                f"      ✔️ processed by {type}: {expansion_str} =>{final_choice} ({resolved_aa})  \n"
                )
                full_log.append(log_entry)
                seq_list[i] = final_choice

                # Track ambiguous positions that remain unresolved
                if type == "remaining":
                    remaining_logs_amb.append(
                        f"[{seq_id}] \n "
                        f"  Original codon {codon_str} (pos: {codon_index+1} ({start+1}-{end})) \n"
                        f"  Dist_con={d_triplet}, Phylo_con={p_triplet}, Global_con={g_triplet}  \n "
                        f"  ❌ not resolved: {expansion_str} =>{final_choice} ({resolved_aa}) ;  Type: {type}) \n"
                    )

                
        # Append the complete sequence log to `full_logs`
        if len(full_log) > 1:  # Ensure at least one resolved codon is logged
            full_logs_amb.append("\n".join(full_log))

        # Final update of sequence
        resolved_sequences[seq_id] = "".join(seq_list)
    print("\n All sequences have been processed for resolving ambiguous bases.\n")
    
    # === Step 2: Resolve single-Ns ===
    
    print("==================== Resolving single-Ns ====================")
    
    global_idx=0
    
    # Process clusters in sorted order:
    cluster_groups_N = {}  # To store modified sequences per cluster

    full_logs_single = []
    remaining_logs_single = []
    for cluster_index in sorted(dist_cluster_groups.keys()):
        cluster_sequences = dist_cluster_groups[cluster_index]
        consensus_dist = distance_consensus.get(cluster_index, None)
        if consensus_dist is None:
            print(f"⚠️  Skipping Cluster {cluster_index} (No distance consensus found).")
            continue

        processed_sequences = []  # To store modified sequences for this cluster

        for seq_id, sequence in cluster_sequences:
            global_idx += 1
            print(f"    [{global_idx}/{total_seqs}] Processing {seq_id}...", end='\r')
            phylo_cluster_id = phylo_membership.get(seq_id)
            if phylo_cluster_id is None:
                print(f"⚠️  Skipping {seq_id} (No phylo membership found).")
                continue
            consensus_phylo = phylo_consensus.get(phylo_cluster_id)
            if consensus_phylo is None:
                print(f"⚠️  Skipping {seq_id} (No phylo consensus found).")
                continue

            # Get final sequence (after ambiguity resolution if already done)
            final_seq = resolved_sequences.get(seq_id, sequence)

            # Apply N resolution
            repaired_seq, vote_log = FN.resolve_single_N_vote(
                seq=final_seq,
                consensus_dist=consensus_dist,
                consensus_phylo=consensus_phylo,
                global_consensus=global_consensus,
                codon_table=FN.codon_table,
                user=args.resolution_strategy,
                mode=args.resolution_mode
            )

            processed_sequences.append((seq_id, repaired_seq))

            # Logging
            if vote_log:

                FN.generate_logs_for_single_N_vote(
                    seq_id=seq_id,
                    original_seq=final_seq,
                    repaired_seq=repaired_seq,
                    vote_log=vote_log,
                    consensus_dist=consensus_dist,
                    consensus_phylo=consensus_phylo,
                    global_consensus=global_consensus,
                    user=args.resolution_strategy,
                    codon_table=FN.codon_table,
                    full_logs=full_logs_single,
                    remaining_logs=remaining_logs_single,
                    cluster_id= cluster_id
                )
                
        cluster_groups_N[cluster_index] = processed_sequences
    
    print("\n All sequences have been processed for resolving single-Ns.\n")
        

    # === Step 3: Resolve multiple-Ns ===
    
    print("==================== Resolving multiple-Ns ====================")
    cluster_groups_adj = {}  # Stores modified sequences per cluster (after gap replacement)
    log_all_clusters_adj = {}  # Stores logs for each cluster

    total_clusters = len(cluster_groups_N)  # Total clusters from your previous step
    global_idx = 0
    
    full_logs_adj = []
    remaining_logs_adj = []
    for cluster_idx, (cluster_index, cluster_sequences) in enumerate(cluster_groups_N.items(), start=1):
        # Use distance_consensus as the consensus for gap replacement.
        consensus_seq = distance_consensus.get(cluster_index, None)
        
        if consensus_seq is None:
            print(f"Skipping Cluster {cluster_index} (No consensus sequence found).")
            continue

        processed_sequences = []  # Store modified sequences for this cluster
        log_per_cluster_adj = []  # Store logs for this cluster

        for seq_index, (seq_id, all_repaired_seqs) in enumerate(cluster_sequences):
            global_idx += 1
            print(f"    [{global_idx}/{total_seqs}] Processing {seq_id}...", end='\r')
            output = FN.replace_multiple_N(all_repaired_seqs, consensus_seq)
            repaired_seq = "".join(output["triplets"])  # Join triplets to form full sequence
            processed_sequences.append((seq_id, repaired_seq))
            
            if output["log"]:
                FN.generate_logs_for_multiple_N_vote(
                    seq_id=seq_id,
                    original_seq=all_repaired_seqs,
                    repaired_seq=repaired_seq,
                    vote_log=output["log"],
                    full_logs=full_logs_adj,
                    remaining_logs=remaining_logs_adj,  # Optional, depending on your logic
                    cluster_id=cluster_index
                )

        cluster_groups_adj[cluster_index] = processed_sequences
        
        
    # Collect all repaired sequences from all clusters.
    all_repaired_seqs = []
    for cluster_id, seq_list in cluster_groups_adj.items():
        all_repaired_seqs.extend(seq_list)

    output_fasta = f"{args.out_dir}{args.out_name}_{args.resolution_strategy}_{args.resolution_mode}_FINALIGN.fasta" # Modify path if needed
    FN.export_to_fasta(all_repaired_seqs, output_fasta)
    
    print(f"\nResolved sequences successfully exported to {output_fasta}!")


    # === Define all log file paths ===
    log_paths = {
        "full_amb":      f"{args.out_dir}{args.out_name}_log_full_amb.log",
        "full_single":   f"{args.out_dir}{args.out_name}_log_full_single.log",
        "full_multi":      f"{args.out_dir}{args.out_name}_log_full_multi.log",
        "unsolved_amb":     f"{args.out_dir}{args.out_name}_log_remaining_amb.log",
        "unsolved_single":  f"{args.out_dir}{args.out_name}_log_remaining_single.log",
        "unsolved_multi":   f"{args.out_dir}{args.out_name}_log_remaining_multi.log"
    }

    # === Write all logs ===
    FN.write_log(log_paths["full_amb"], full_logs_amb)
    FN.write_log(log_paths["unsolved_amb"], remaining_logs_amb)

    FN.write_log(log_paths["full_single"], full_logs_single)
    FN.write_log(log_paths["unsolved_single"], remaining_logs_single)

    FN.write_log(log_paths["full_multi"], full_logs_adj)
    FN.write_log(log_paths["unsolved_multi"], remaining_logs_adj)

    # === Final message ===
    print(f"\n✅ Resolved sequences exported to: {output_fasta}")
    print("📝 Logs saved to:")
    for label, path in log_paths.items():
        print(f"  - {label}: {path}")

    

if __name__ == "__main__":
    main()
