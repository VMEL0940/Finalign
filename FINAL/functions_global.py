# %%
import argparse
import csv
import itertools
import logging
import math
import os
import pathlib
import random
import re
import string
import subprocess
import sys
import tempfile
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from io import StringIO
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy.random import default_rng

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score, pairwise_distances
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from patsy import dmatrices
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colors, cm
import seaborn as sns
from logomaker import Logo
from PIL import Image
from Bio import SeqIO, Phylo
from Bio.Seq import Seq, translate
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import seq3
from Bio.Data import CodonTable

from treeswift import read_tree_newick
from ete3 import Tree
#from ete3 import Tree, TreeStyle, NodeStyle, TextFace

from tqdm import tqdm
import shutil
import subprocess

# %%
# Append the path to the local TreeCluster folder
#sys.path.append("./TreeCluster-master")  # Adjust if needed
sys.path.append("./TreeCluster-master")
import TreeCluster

# %% [markdown]
# ### Base Functions

# %%
def clean_sequences_with_ids(input_fasta, threshold):
    """
    Reads sequences from a FASTA file, cleans them, and returns a list of tuples containing IDs and cleaned sequences.
    Sequences with more than the threshold percentage of 'N' are removed.
    """
    valid_bases = set('ACGTYRSWKMBDHV')
    cleaned_sequences = []

    def clean_sequence(sequence, valid_bases):
        sequence = sequence.upper()
        return ''.join([base if base in valid_bases else 'N' for base in sequence])

    for record in SeqIO.parse(input_fasta, 'fasta'):
        cleaned_seq = clean_sequence(str(record.seq), valid_bases)
        if cleaned_seq.count('N') / len(cleaned_seq) <= threshold:
            cleaned_sequences.append((record.id, cleaned_seq))

    return cleaned_sequences

def cons_matrix(cluster_sequences):
    """
    Given a list of sequences from a cluster, generate a consensus sequence.
    """
    if not cluster_sequences:
        return ""
    
    # Ensure all sequences are of the same length
    sequence_length = len(cluster_sequences[0])
    
    # Initialize an empty list to hold the consensus sequence
    consensus_sequence = []
    
    # Iterate over each position in the sequence
    for i in range(sequence_length):
        # Get the nucleotides at the current position for all sequences
        column_bases = [seq[i] for seq in cluster_sequences if len(seq) > i]
        
        # Find the most common base at this position
        base_counter = Counter(column_bases)
        most_common_base, _ = base_counter.most_common(1)[0]
        
        # Append the most common base to the consensus sequence
        consensus_sequence.append(most_common_base)
    
    # Return the consensus sequence as a string
    return ''.join(consensus_sequence)

def export_to_fasta(sequences_with_ids, output_file):
    """Export sequences with IDs to a FASTA file (80 chars per line)."""
    with open(output_file, 'w') as fasta_file:
        for seq_id, seq in sequences_with_ids:
            fasta_file.write(f">{seq_id}\n")
            for i in range(0, len(seq), 80):
                fasta_file.write(seq[i:i+80] + "\n")
                
def write_log(filename, log_data):
    """Writes log data to a file."""
    with open(filename, "w") as f:
        for line in log_data:
            f.write(line + "\n")

# %% [markdown]
# ### Distance-based Clustering (TARDiS)

# %%
# Jukes-Cantor distance calculation
def jukes_cantor_distance(seq1, seq2):
    p = sum(a != b for a, b in zip(seq1, seq2)) / len(seq1)
    return -0.75 * math.log(1 - (4 / 3) * p) if 0 < p < 0.75 else float('inf') if p >= 0.75 else 0.0

def compute_distance_matrix(alignment):
    labels = [record.id for record in alignment]
    n = len(labels)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = jukes_cantor_distance(str(alignment[i].seq), str(alignment[j].seq))
            matrix[i, j] = matrix[j, i] = d
    return matrix, labels

# Genetic diversity fitness calculation (Fgd)
def calc_fgd(indices, dist_matrix):
    pairs = list(itertools.combinations(indices, 2))
    total = sum(dist_matrix[i, j] for i, j in pairs)
    max_possible = sum(sorted(dist_matrix[np.triu_indices_from(dist_matrix, 1)])[-len(pairs):])
    return total / max_possible if max_possible > 0 else 0

# Temporal distribution fitness calculation (Ftd)
def calc_ftd(indices, dates):
    sampled_dates = sorted(dates[i] for i in indices)
    ideal_dates = np.linspace(min(dates), max(dates), len(indices))
    worst_dates = [np.mean(sampled_dates)] * len(indices)
    numerator = sum(abs(s - i) for s, i in zip(sampled_dates, ideal_dates))
    denominator = sum(abs(w - i) for w, i in zip(worst_dates, ideal_dates))
    return 1 - numerator / denominator if denominator > 0 else 1

# Combined fitness
def calc_fitness(indices, dist_matrix, dates, wgd=0.5, wtd=0.5):
    return wgd * calc_fgd(indices, dist_matrix) + wtd * calc_ftd(indices, dates)

# Crossover function based on TARDiS
def crossover(a, b, n):
    shared = list(set(a) & set(b))
    unique = list(set(a) ^ set(b))
    if len(shared) >= n:
        return random.sample(shared, n)
    return shared + random.sample(unique, n - len(shared))

# Mutation function based on TARDiS
def mutate(individual, pool, rate=0.08):
    if random.random() < rate:
        idx = random.randint(0, len(individual) - 1)
        replacements = list(set(pool) - set(individual))
        if replacements:
            individual[idx] = random.choice(replacements)
    return individual

# Genetic algorithm implementation based on TARDiS
def evolve_cluster(dist_matrix, dates, pool, n, generations=50, pop_size=100, elite_frac=0.05, random_frac=0.10, wgd=0.5, wtd=0.5):
    population = [random.sample(pool, n) for _ in range(pop_size)]
    for gen in range(generations):
        fitness_scores = [calc_fitness(ind, dist_matrix, dates, wgd, wtd) for ind in population]
        sorted_pop = [ind for _, ind in sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)]

        elite_n = int(pop_size * elite_frac)
        random_n = int(pop_size * random_frac)
        evolved_n = pop_size - elite_n - random_n

        new_population = sorted_pop[:elite_n]

        # Tournament selection for crossover
        while len(new_population) < elite_n + evolved_n:
            tournament_a = max(random.sample(population, 5), key=lambda ind: calc_fitness(ind, dist_matrix, dates, wgd, wtd))
            tournament_b = max(random.sample(population, 5), key=lambda ind: calc_fitness(ind, dist_matrix, dates, wgd, wtd))
            child = crossover(tournament_a, tournament_b, n)
            child = mutate(child, pool)
            new_population.append(child)

        # Randomly add new individuals
        while len(new_population) < pop_size:
            new_population.append(random.sample(pool, n))

        population = new_population

    final_scores = [calc_fitness(ind, dist_matrix, dates, wgd, wtd) for ind in population]
    best_idx = np.argmax(final_scores)
    return population[best_idx]

# Cluster-based subsampling implementation (TARDiS)
def cluster_subsampling(dist_matrix, labels, dates, n_clusters=5, sample_size=1, wgd=0.5, wtd=0.5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(dist_matrix)
    representatives = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_ids == cluster_id)[0]
        sub_dist_matrix = dist_matrix[np.ix_(cluster_indices, cluster_indices)]
        sub_dates = [dates[i] for i in cluster_indices]
        best = evolve_cluster(sub_dist_matrix, sub_dates, list(range(len(cluster_indices))),
                              n=sample_size, wgd=wgd, wtd=wtd)
        representatives.extend([labels[cluster_indices[idx]] for idx in best])
    return representatives

# Assign all sequences to nearest representative
def assign_to_representatives(dist_matrix, labels, representatives):
    rep_indices = [labels.index(rep) for rep in representatives]
    clusters = {rep: [rep] for rep in representatives}  # <-- fix here (include representative itself)
    for i, label in enumerate(labels):
        if label in representatives:
            continue  # already assigned
        closest_rep = representatives[np.argmin([dist_matrix[i, r] for r in rep_indices])]
        clusters[closest_rep].append(label)
    return clusters

# Match all id with sequence 
def sort_sequences_into_representative_clusters(sequences_with_ids, clusters):
    """
    Sorts sequences into clusters based on a representative-based clustering dictionary.

    Parameters:
        sequences_with_ids (list): List of tuples where each tuple contains (ID, sequence).
        clusters (dict): Dictionary where each key is a representative ID, and values are lists of sequence IDs.

    Returns:
        dict: A dictionary where each key is a representative ID, and the value is a list of (ID, sequence) tuples
              belonging to that representative's cluster.
    """
    # Convert sequence list to dictionary for fast lookup
    seq_dict = {seq_id: seq for seq_id, seq in sequences_with_ids}

    sorted_clusters = {}
    for rep_id, seq_id_list in clusters.items():
        sorted_clusters[rep_id] = []
        for seq_id in seq_id_list:
            seq = seq_dict.get(seq_id)
            if seq is not None:
                sorted_clusters[rep_id].append((seq_id, seq))
            else:
                print(f"Warning: Sequence ID '{seq_id}' not found in provided sequences.")
    return sorted_clusters

def subsample_and_assign_clusters(dist_matrix, labels, dates, n_clusters=5, sample_size=1, wgd=0.5, wtd=0.5):
    
    print(f"Weight of genetic distance: {wgd}; Weight of time: {wtd}.")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(dist_matrix)

    clusters = {}
    rep_to_members = {}

    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_ids == cluster_id)[0]
        sub_dist_matrix = dist_matrix[np.ix_(cluster_indices, cluster_indices)]
        sub_dates = [dates[i] for i in cluster_indices]

        best = evolve_cluster(
            sub_dist_matrix, sub_dates,
            list(range(len(cluster_indices))),
            n=sample_size, wgd=wgd, wtd=wtd
        )

        # Determine representatives in original label space
        cluster_reps = [labels[cluster_indices[idx]] for idx in best]
        for rep in cluster_reps:
            clusters[rep] = [rep]  # Initialize with self
            rep_to_members[rep] = cluster_indices  # Store indices for this cluster

    # Assign all sequences to the nearest representative
    rep_indices = [labels.index(rep) for rep in clusters.keys()]
    for i, label in enumerate(labels):
        if label in clusters:
            continue  # Already added
        distances = [dist_matrix[i, labels.index(rep)] for rep in clusters]
        closest_rep = list(clusters.keys())[np.argmin(distances)]
        clusters[closest_rep].append(label)

    return clusters


# %% [markdown]
# ### Phylogenetic-based Clustering (TreeCluster)

# %%
def convert_non_ACGT_to_gaps(seq_id_pairs):
    """
    Given a list of (seq_id, seq_str),
    return a list of (seq_id, seq_str_with_gaps) where non-ACGT characters are replaced with '-'.
    """
    sanitized = []
    for seq_id, seq_str in seq_id_pairs:
        new_seq = ''.join(ch if ch in "ACGT" else '-' for ch in seq_str)
        sanitized.append((seq_id, new_seq))
    return sanitized

def write_sequences_to_fasta(seq_id_pairs, output_path):
    """
    Save a list of (id, seq) pairs into a FASTA file.
    """
    with open(output_path, "w") as f:
        for seq_id, sequence in seq_id_pairs:
            f.write(f">{seq_id}\n{sequence}\n")

def run_fasttree(fasta_path, output_nwk, is_nt=True):
    """
    Calls the FastTree binary (found in PATH) on a FASTA file and saves the tree to output_nwk.
    Automatically detects the FastTree path from system environment.
    """
    fasttree_path = shutil.which("FastTree")
    if fasttree_path is None:
        raise FileNotFoundError("❌ FastTree not found in PATH. Please install or link it to a directory like /usr/local/bin")

    cmd = [fasttree_path]
    if is_nt:
        cmd.append("-nt")
    cmd.append(fasta_path)

    with open(output_nwk, "w") as outfile:
        subprocess.run(cmd, stdout=outfile)

    print(f"✅ Tree written to: {output_nwk}")


def cluster_tree_with_treecluster(newick_path, threshold=0.03, support=0.0, method="max_clade"):
    # Read Newick string
    with open(newick_path) as f:
        newick_str = f.read().strip()

    # Load tree with TreeCluster
    tree = TreeCluster.read_tree_newick(newick_str)
    
    # Also parse it with Bio.Phylo for traversals
    handle = StringIO(newick_str)
    phylo_tree = Phylo.read(handle, "newick")

    # Perform clustering
    clusters = TreeCluster.METHODS[method](tree, threshold, support)

    leaf_to_cluster = {}
    new_clusters = []
    for cluster in clusters:
        if len(cluster) > 1:
            new_clusters.append(cluster)
            for leaf in cluster:
                leaf_to_cluster[leaf] = cluster

    # Get singleton leaves
    singleton_leaves = [cluster[0] for cluster in clusters if len(cluster) == 1]

    # Build name-to-node map for Bio.Phylo
    name_to_terminal = {leaf.name: leaf for leaf in phylo_tree.get_terminals()}

    for singleton in singleton_leaves:
        node = name_to_terminal[singleton]
        closest = None
        min_dist = float('inf')

        for other_leaf in phylo_tree.get_terminals():
            if other_leaf.name == singleton or other_leaf.name in singleton_leaves:
                continue

            dist = phylo_tree.distance(node, other_leaf)
            if dist < min_dist:
                closest = other_leaf
                min_dist = dist

        if closest and closest.name in leaf_to_cluster:
            leaf_to_cluster[closest.name].append(singleton)

    return new_clusters

def build_fasttree_and_sweep_thresholds(
    cleaned_sequences_with_ids,
    threshold_range=(0.03, 0.1, 0.001),  # (start, end, step)
    treecluster_support=0.0,
    treecluster_method="max_clade",
    nwk_name="None",
    nwk_path="./results/"
):
    if not os.path.exists(nwk_path):
        os.makedirs(nwk_path)

    tmp_mapping = {}
    tmp_seq_pairs = []
    for i, (orig_id, seq) in enumerate(cleaned_sequences_with_ids, 1):
        tmp_id = f"tmp{i}"
        tmp_mapping[tmp_id] = orig_id
        tmp_seq_pairs.append((tmp_id, seq))

    fasta_path = os.path.join(nwk_path, nwk_name + "_input.fasta")
    newick_path = os.path.join(nwk_path, nwk_name + "_tree.nwk")
    write_sequences_to_fasta(tmp_seq_pairs, fasta_path)

    run_fasttree(fasta_path, newick_path)

    # Generate thresholds using np.arange
    thresholds = np.arange(*threshold_range)

    threshold_results = {}
    for threshold in thresholds:
        tmp_clusters = cluster_tree_with_treecluster(
            newick_path=newick_path,
            threshold=threshold,
            support=treecluster_support,
            method=treecluster_method
        )

        restored_clusters = []
        for cluster in tmp_clusters:
            restored = [tmp_mapping[tid] for tid in cluster]
            restored_clusters.append(restored)

        cluster_dict = {i + 1: cluster for i, cluster in enumerate(restored_clusters)}
        threshold_results[round(threshold, 5)] = cluster_dict
        #print(f"Threshold {threshold:.3f} → {len(cluster_dict)} clusters")

    return threshold_results, newick_path, tmp_mapping

def relabel_newick_tree(tmp_newick_path, tmp_mapping, output_newick_path):
    """
    Relabel a Newick tree built with tmp1/tmp2... back to original names.
    """
    tree = Tree(tmp_newick_path, format=1)
    for leaf in tree.iter_leaves():
        if leaf.name in tmp_mapping:
            leaf.name = tmp_mapping[leaf.name]
    tree.write(outfile=output_newick_path, format=1)

def plot_elbow_point(clusters_by_threshold):
    # Remove thresholds where clustering resulted in only 1 cluster
    clusters_by_threshold = {
        t: clusters for t, clusters in clusters_by_threshold.items() if len(clusters) > 1
    }

    sorted_thresholds = sorted(clusters_by_threshold.keys())
    cluster_counts = [len(clusters_by_threshold[t]) for t in sorted_thresholds]

    drops = [cluster_counts[i - 1] - cluster_counts[i] for i in range(1, len(cluster_counts))]
    max_drop_idx = drops.index(max(drops)) + 1
    elbow_threshold = sorted_thresholds[max_drop_idx]
    elbow_cluster_count = cluster_counts[max_drop_idx]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_thresholds, cluster_counts, marker='o', label='Cluster count')
    plt.axvline(x=elbow_threshold, color='red', linestyle='--', label=f'Elbow: {elbow_threshold:.3f}')
    plt.scatter([elbow_threshold], [elbow_cluster_count], color='red', s=100, zorder=5)

    plt.title("Elbow Detection: Threshold vs Number of Clusters")
    plt.xlabel("TreeCluster Threshold")
    plt.ylabel("Number of Clusters")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return elbow_threshold

def detect_elbow_threshold(clusters_by_threshold):
    
    # Remove thresholds that resulted in only 1 cluster
    filtered = {t: c for t, c in clusters_by_threshold.items() if len(c) > 1}

    if len(filtered) < 2:
        raise ValueError("Not enough thresholds with >1 clusters to detect an elbow point.")

    sorted_thresholds = sorted(filtered.keys())
    cluster_counts = [len(filtered[t]) for t in sorted_thresholds]

    # Calculate the drop between successive thresholds
    drops = [cluster_counts[i - 1] - cluster_counts[i] for i in range(1, len(cluster_counts))]
    max_drop_idx = drops.index(max(drops)) + 1  # +1 because drop starts from index 1
    elbow_threshold = sorted_thresholds[max_drop_idx]

    return elbow_threshold


def sort_sequences_into_clusters(sequences_with_ids, cluster_dict):
    """
    Sort sequences into cluster groups based on TreeCluster results.

    Parameters:
        sequences_with_ids: List of (seq_id, sequence)
        cluster_dict: Dict of {cluster_id: [seq_ids]}

    Returns:
        Dict of {cluster_id: [(seq_id, sequence)]}
    """
    # Build quick lookup for ID → sequence
    id_to_seq = {seq_id: seq_str for seq_id, seq_str in sequences_with_ids}

    cluster_groups = {}
    for cluster_id, id_list in cluster_dict.items():
        cluster_groups[cluster_id] = [(seq_id, id_to_seq[seq_id]) for seq_id in id_list if seq_id in id_to_seq]
    return cluster_groups


# %% [markdown]
# ### Main Functions for Resolving

# %%
###########################################################################
# STEP 1: TRIMMING AND CLEANING SEQUENCES
###########################################################################

def probable_start(seqs):
    start_codon = "ATG"
    start_codon_positions = []
    for seq_id, seq in seqs:
        match = re.search(start_codon, str(seq))
        if match:
            # +1 for 1-based indexing
            start_codon_positions.append(match.start() + 1)
        else:
            start_codon_positions.append(None)
    filtered_positions = [pos for pos in start_codon_positions if pos is not None]
    if filtered_positions:
        most_frequent_position = Counter(filtered_positions).most_common(1)[0][0]
        return most_frequent_position
    return None

def probable_stop(seqs):
    stop_codons = ["TAA", "TAG", "TGA"]
    # Replace 'N' with '-' to avoid interference
    seqs_modified = [(seq_id, str(seq).replace('N', '-')) for seq_id, seq in seqs]
    max_length = max(len(seq) for _, seq in seqs_modified)
    stop_codon_positions = []
    for seq_id, seq in seqs_modified:
        positions = []
        for stop_codon in stop_codons:
            positions += [m.start() + 1 for m in re.finditer(stop_codon, seq)]
        if positions:
            # Get the last stop codon position
            stop_codon_positions.append(max(positions))
    if stop_codon_positions:
        most_frequent_stop_pos = Counter(stop_codon_positions).most_common(1)[0][0]
        # +2 adjusts to include the full codon (1-based)
        return most_frequent_stop_pos + 2
    return max_length  # Fallback: end of sequence

def cut_sequences(sequences_with_ids):
    start_pos = probable_start(sequences_with_ids)
    stop_pos = probable_stop(sequences_with_ids)
    if start_pos is None or stop_pos is None:
        print("No valid start or stop position found.")
        return []
    cut_sequences = []
    for seq_id, seq in sequences_with_ids:
        if len(seq) >= stop_pos:
            cut_seq = seq[start_pos - 1: stop_pos]  # convert to 0-indexed slice
            cut_sequences.append((seq_id, cut_seq))
        else:
            cut_sequences.append((seq_id, ""))
    return cut_sequences

def remove_gap_columns(seqs):
    if not seqs:
        return []
    seq_list = [list(str(seq[1])) for seq in seqs]
    seq_length = len(seq_list[0])
    columns_to_keep = []
    deleted_positions = []
    for i in range(seq_length):
        column = [seq[i] for seq in seq_list]
        gap_fraction = column.count('-') / len(column)
        if gap_fraction < 0.99:
            columns_to_keep.append(i)
        else:
            deleted_positions.append(i + 1)  # record 1-indexed
    cleaned_seqs = []
    for seq_id, seq in seqs:
        cleaned_seq = ''.join([seq[i] for i in columns_to_keep])
        cleaned_seqs.append((seq_id, cleaned_seq))
    print(f"Deleted positions (1-indexed): {deleted_positions}")
    return cleaned_seqs


# %%
###########################################################################
# STEP 2: AMBIGUOUS BASE RESOLUTION (USING DISTANCE, PHYLO, & GLOBAL CONSENSUS)
###########################################################################

# Custom IUPAC map and helper functions
iupac_map = {
    'R': {'A','G'},
    'Y': {'C','T'},
    'S': {'G','C'},
    'W': {'A','T'},
    'K': {'G','T'},
    'M': {'A','C'},
    'B': {'C','G','T'},
    'D': {'A','G','T'},
    'H': {'A','C','T'},
    'V': {'A','C','G'},
}

def show_expansion(base, iupac_map):
    if base in iupac_map:
        expansions = ",".join(sorted(iupac_map[base]))
        return f"{base}{{{expansions}}}"
    else:
        return base


# Standard codon table remains unchanged
codon_table = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

def resolve_ambiguous(amb, d, p, g, iupac_map, codon_table, codon_str, user):
    """
    Resolve ambiguous nucleotide using distance (d), phylogeny (p), and global (g) preferences.

    Returns:
        (str, str, str, str, str, str, str): 
        (Resolved nucleotide, original amino acid, new amino acid, d_aa, p_aa, g_aa, resolution type)
    """
    allowed_bases = set(iupac_map.get(amb, []))
    in_d, in_p, in_g = d in allowed_bases, p in allowed_bases, g in allowed_bases

    # Generate possible codons and amino acids
    possible_codons_dict = {key: codon_str.replace(amb, key) for key in {d, p, g} if key in allowed_bases}
    aa_dict = {key: codon_table.get(value, None) for key, value in possible_codons_dict.items()}

    d_aa, p_aa, g_aa = aa_dict.get(d), aa_dict.get(p), aa_dict.get(g)

    # Case 1: d and p match
    if in_d and in_p and d == p:
        return d, d_aa, d_aa, p_aa, g_aa, "silent"

    # Case 2: d_aa and p_aa are identical
    if in_d and in_p and d_aa == p_aa:
        if g == d:
            return d, d_aa, d_aa, p_aa, g_aa, "silent"
        elif g == p:
            return p, p_aa, d_aa, p_aa, g_aa, "silent"
        else:
            if user == "g" and in_g:
                return g, g_aa, d_aa, p_aa, g_aa, "silent"
            elif user == "d":
                return d, d_aa, d_aa, p_aa, g_aa, "silent"
            elif user == "p":
                return p, p_aa, d_aa, p_aa, g_aa, "silent"

    # Case 3: d_aa and p_aa differ
    if in_d and in_p and d_aa != p_aa:
        if g == d or g == p:
            return g, g_aa, d_aa, p_aa, g_aa, "nonsilent"
        else:
            if user == "g" and in_g:
                return g, g_aa, d_aa, p_aa, g_aa, "nonsilent"
            elif user == "d":
                return d, d_aa, d_aa, p_aa, g_aa, "nonsilent"
            elif user == "p":
                return p, p_aa, d_aa, p_aa, g_aa, "nonsilent"

    # Case 4: Only d is valid
    if in_d and not in_p:
        if d == g:
            return d, d_aa, d_aa, p_aa, g_aa, "silent"
        elif user == "d":
            return d, d_aa, d_aa, p_aa, g_aa, "personalized"
        elif user == "g" and in_g:
            return g, g_aa, d_aa, p_aa, g_aa, "personalized"

    # Case 5: Only p is valid
    if in_p and not in_d:
        if p == g:
            return p, p_aa, d_aa, p_aa, g_aa, "silent"
        elif user == "p":
            return p, p_aa, d_aa, p_aa, g_aa, "personalized"
        elif user == "g" and in_g:
            return g, g_aa, d_aa, p_aa, g_aa, "personalized"

    # Final fallback
    return amb, None, d_aa, p_aa, g_aa, "remaining"



# %%

# Build: amino acid → list of codons

aa_to_codons = defaultdict(list)
for codon, aa in codon_table.items():
    aa_to_codons[aa].append(codon)

def resolve_single_N_vote(seq, consensus_dist, consensus_phylo, global_consensus, codon_table, user, mode):
    """
    Resolve isolated 'N' bases using distance, phylo, and global consensus.

    Args:
        seq (str): Nucleotide sequence with possible isolated 'N'.
        consensus_dist, consensus_phylo, global_consensus (str): Consensus sequences.
        codon_table (dict): Codon-to-amino-acid dictionary.
        user (str): Priority source when ambiguity remains ("d", "p", or "g").
        mode (str): Mode for fallback behavior (e.g., "clean" allows user-based fallback).

    Returns:
        repaired_seq (str): Sequence with resolved Ns.
        vote_log (list of tuples): List of
            (resolved_nt, original_aa, new_aa, d_aa, p_aa, g_aa, resolution_type)
    """
    vote_log = []
    seq = list(seq)

    # Identify isolated Ns
    gap_positions = [
        i for i, base in enumerate(seq)
        if base == "N" and i > 0 and i < len(seq) - 1 and seq[i - 1] in "ACGT" and seq[i + 1] in "ACGT"
    ]

    for pos in gap_positions:
        if pos >= len(consensus_dist) or pos >= len(consensus_phylo) or pos >= len(global_consensus):
            continue

        codon_index = pos // 3
        start = codon_index * 3
        end = start + 3
        if end > len(seq):
            continue

        codon = seq[start:end]
        codon_str = ''.join(codon)

        d_codon = consensus_dist[start:end]
        p_codon = consensus_phylo[start:end]
        g_codon = global_consensus[start:end]

        d_aa = codon_table.get(d_codon)
        p_aa = codon_table.get(p_codon)
        g_aa = codon_table.get(g_codon)

        d_base = consensus_dist[pos]
        p_base = consensus_phylo[pos]
        g_base = global_consensus[pos]

        resolved_nt = None
        resolution_type = "remaining"

        # Case 1: d and p agree on AA
        if d_aa == p_aa:
            if d_base == p_base:
                resolved_nt = d_base
                resolution_type = "silent"
            elif g_base == d_base:
                resolved_nt = d_base
                resolution_type = "silent"
            elif g_base == p_base:
                resolved_nt = p_base
                resolution_type = "silent"
            else:
                if user == "d":
                    resolved_nt = d_base
                elif user == "p":
                    resolved_nt = p_base
                elif user == "g" and g_aa == d_aa == p_aa:
                    resolved_nt = g_base
                resolution_type = "personalized"

        # Case 2: d and p differ in AA
        elif d_aa != p_aa:
            if g_base == d_base:
                resolved_nt = d_base
                resolution_type = "nonsilent"
            elif g_base == p_base:
                resolved_nt = p_base
                resolution_type = "nonsilent"
            else:
                if g_aa == d_aa:
                    if user == "d":
                        resolved_nt = d_base
                        resolution_type = "nonsilent"
                    elif user == "g":
                        resolved_nt = g_base
                        resolution_type = "nonsilent"
                    elif user == "p" and mode == "clean":
                        resolved_nt = p_base
                        resolution_type = "personalized"

                elif g_aa == p_aa:
                    if user == "p":
                        resolved_nt = p_base
                        resolution_type = "nonsilent"
                    elif user == "g":
                        resolved_nt = g_base
                        resolution_type = "nonsilent"
                    elif user == "d" and mode == "clean":
                        resolved_nt = d_base
                        resolution_type = "personalized"

                else:  # all different
                    if mode == "clean":
                        if user == "d":
                            resolved_nt = d_base
                        elif user == "p":
                            resolved_nt = p_base
                        elif user == "g":
                            resolved_nt = g_base
                        resolution_type = "personalized"

        # Final fallback
        if resolved_nt not in "ACGT":
            vote_log.append(("N", None, None, d_aa, p_aa, g_aa, "remaining"))
            continue

        # Apply resolution
        seq[pos] = resolved_nt
        codon[pos % 3] = resolved_nt
        new_codon = ''.join(codon)
        new_aa = codon_table.get(new_codon, "-")
        original_aa = codon_table.get(codon_str, None)

        vote_log.append((pos, resolved_nt, original_aa, new_aa, d_aa, p_aa, g_aa, resolution_type))

    repaired_seq = ''.join(seq)
    return repaired_seq, vote_log

def generate_logs_for_single_N_vote(
    seq_id, original_seq, repaired_seq, vote_log,
    consensus_dist, consensus_phylo, global_consensus,
    user, codon_table,
    full_logs, remaining_logs,
    cluster_id
):
    """
    Generate logs from resolve_single_N_vote and append to provided log lists.
    """
    full_log = [f"[{seq_id}]"]

    for pos, resolved_nt, original_aa, new_aa, d_aa, p_aa, g_aa, res_type in vote_log:
        
        codon_index = pos // 3
        start = codon_index * 3
        end = start + 3

        codon_str = original_seq[start:end]
        new_codon_str = repaired_seq[start:end]

        d_triplet = f"{consensus_dist[start:end]} ({d_aa if d_aa else '-'})"
        p_triplet = f"{consensus_phylo[start:end]} ({p_aa if p_aa else '-'})"
        g_triplet = f"{global_consensus[start:end]} ({g_aa if g_aa else '-'})"

        expansion_str = "N"
        final_choice = resolved_nt
        resolved_aa = new_aa
        codon_str_display = "".join(codon_str)

        full_log.append(
            f"   Original codon {codon_str_display} (pos: {codon_index+1} ({start+1}-{end})) \n"
            f"   Dist_con={d_triplet}, Phylo_con={p_triplet}, Global_con={g_triplet}  \n"
            f"      ✔️ processed by {res_type}: {expansion_str} =>{final_choice} ({resolved_aa})  \n"
        )

        log_entry = (
            f"[{seq_id}] \n"
            f"  Original codon {codon_str_display} (pos: {codon_index+1} ({start+1}-{end})) \n"
            f"  Dist_con={d_triplet}, Phylo_con={p_triplet}, Global_con={g_triplet}  \n"
        )
        
        if res_type == "remaining":
            remaining_logs.append(log_entry + f"  ❌ not resolved: {expansion_str} =>{final_choice} ({resolved_aa}) ;  Type: {res_type}) \n")

    if len(full_log) > 1:
        full_logs.append(f"[Cluster {cluster_id} - {seq_id}]\n" + "\n".join(full_log[1:]) + "\n")



# %%
def find_nearest_non_zero(index, triplets):
    """Finds the nearest triplet with at least one non-gap base."""
    for n in range(index + 1, len(triplets)):
        if any(base != "-" for base in triplets[n]):  # Check for non-gap characters
            return n
    return None  # Return None if no valid triplet is found

def count_non_gaps(triplet):
    """Counts the number of non-gap bases in a triplet."""
    return sum(1 for base in triplet if base in "ACGT")

def replace_multiple_N(seq, consensus_string):
    """Replaces gaps (converted from N) in sequences based on the consensus sequence.
       This version works on triplets. It first converts N to '-' and then, based on
       the number of non-gap bases in the triplet, attempts a replacement.
    """
    # Replace N with '-' to work with gaps
    seq = seq.replace("N", "-")
    triplets = [seq[i:i+3] for i in range(0, len(seq), 3)]
    consensus_triplets = [consensus_string[i:i+3] for i in range(0, len(consensus_string), 3)]
    
    log = []
    i = 0  # Initialize triplet index

    while i < len(triplets):
        non_gap_count = count_non_gaps(triplets[i])
        
        if non_gap_count in [1, 2]:  # Only process when triplet has 1 or 2 non-gaps
            n = find_nearest_non_zero(i, triplets)
            
            if n is not None:
                non_gap_next = count_non_gaps(triplets[n])
                original_i, original_n = triplets[i], triplets[n]

                if non_gap_count == 1:
                    if non_gap_next == 1:
                        type_change = "1-1 -> 0-0"
                        triplets[i] = "---"
                        triplets[n] = "---"
                    elif non_gap_next == 2:
                        type_change = "1-2 -> 0-3"
                        nucleotide_string = re.findall(r"[ACGT]", triplets[i])[0] + "".join(re.findall(r"[ACGT]", triplets[n])[:2])
                        triplets[i] = "---"
                        triplets[n] = nucleotide_string if nucleotide_string not in ["TAG", "TGA", "TAA"] else consensus_triplets[n]
                    elif non_gap_next == 3:
                        type_change = "1-3 -> 0-3"
                        triplets[i] = "---"
                    else:
                        type_change = "1-X -> 0-X"
                        triplets[i] = "---"

                elif non_gap_count == 2:
                    if non_gap_next == 1:
                        type_change = "2-1 -> 3-0"
                        nucleotide_string = "".join(re.findall(r"[ACGT]", triplets[i])[:2]) + re.findall(r"[ACGT]", triplets[n])[0]
                        triplets[i] = nucleotide_string if nucleotide_string not in ["TAG", "TGA", "TAA"] else consensus_triplets[i]
                        triplets[n] = "---"
                    elif non_gap_next == 2:
                        type_change = "2-2 -> 3-3"
                        triplets[i] = consensus_triplets[i]
                        triplets[n] = consensus_triplets[n]
                    elif non_gap_next == 3:
                        type_change = "2-3 -> 3-3"
                        triplets[i] = consensus_triplets[i]
                    else:
                        type_change = "2-X -> 3-X"
                        triplets[i] = consensus_triplets[i]
                        triplets[n] = consensus_triplets[n]
                
                log.append({
                    "pos": (i, n),
                    "original_triplets": (original_i, original_n),
                    "repaired_triplets": (triplets[i], triplets[n]),
                    "type": type_change
                })

        i += 1  # Move to next triplet
        
    return {"log": log, "triplets": triplets}

def generate_logs_for_multiple_N_vote(
    seq_id, original_seq, repaired_seq, vote_log,
    full_logs, remaining_logs,  # You can still use the same global log lists
    cluster_id
):
    full_log = [f"[{seq_id}]"]

    for entry in vote_log:
        pos_i, pos_n = entry["pos"]
        original_i, original_n = entry["original_triplets"]
        repaired_i, repaired_n = entry["repaired_triplets"]
        type_change = entry["type"]

        full_log.append(
            f"   Affected codons {pos_i+1} ~ {pos_n+1} \n"
            f"      Original: {original_i} ~ {original_n}\n"
            f"      Repaired: {repaired_i} ~ {repaired_n}\n"
            f"      ✔️ Type: {type_change}  \n"
        )

    if len(full_log) > 1:
        full_logs.append(f"[Cluster {cluster_id} - {seq_id}]\n" + "\n".join(full_log[1:]) + "\n")

