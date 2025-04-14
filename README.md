# FINALGIN
Multiple sequence alignment (MSA) analysis of viral genomes is indispensable for investigating mutations and phylogenetic relationships. Although many alignment tools, such as MAFFT, are widely used for aligning viral sequences, their results often require additional refinement to ensure codon-level accuracy, particularly in the presence of ambiguous bases and erroneous gaps. To address this, FINALIGN was developed as a post-alignment refinement tool to resolve ambiguities and correct misalignments in viral MSAs.

# Abstract Figure

![Figure1](https://github.com/user-attachments/assets/0ae0edb2-7b5a-49dd-afbc-bc674bece8e8)

# Quick start
Download FINALGIN and make sure [dependencies](#Dependencies) are installed. For the quickest start, just run our [example](#example).
You can retrieve example data in `/example`.

# Command line
You can run FINALGIN from the command line as follows:

**Note**: Your input should be the result of an alignment tool such as MAFFT, and we strongly recommend trimming your MSA dataset to start with an **AUG** codon and end with valid stop codons **(UGA, UAA, UAG)**.

Run ``` python Finalign_CLI.py -h ```  to display all available options. 

```
usage: Finalign_CLI.py [-h] --input_fasta INPUT_FASTA --metadata_csv METADATA_CSV
                       [--date {YMD,Y}] --out_dir OUT_DIR --out_name OUT_NAME
                       [--resolution_strategy {d,p,g}] [--resolution_mode {clean,raw}]
                       [--proceed_trimming {yes,no}] [--n_threshold N_THRESHOLD]
                       [--wgd WGD]

🧬 FINALIGN: Resolve your MSA for analysis

optional arguments:
  -h, --help                        : show this help message and exit
  --input_fasta INPUT_FASTA         : Path to input MSA FASTA file
  --metadata_csv METADATA_CSV       : Path to metadata CSV file
  --date {YMD,Y}                    : If using full date {YMD} or using only Year {Y}
  --out_dir OUT_DIR                 : Directory to save output files
  --out_name OUT_NAME`              : Name of output files
  --resolutionary_strategy {d,p,g}  : Preferred resolution strategy
                                        {d} = Distance-based consensus
                                        {p} = Phylogenetic-based consensus (default)
                                        {g} = Global MSA consensus
  --resolution_mode {clean,raw}     : Fallback behavior for single N
                                        clean = Applies fallback strategy when consensus types disagree (default)
                                        raw = Applies resolution only when all consensus sources agree
  --proceed_trimming {yes,no}       : Trim sequences to ORF region based on start/stop codons (default: no)
  --n_threshold N_THRESHOLD         : Maximum allowed proportion of Ns (Default: 0.02)
  --wgd WGD                         : Weight of genetic distance (0–1 (Default: 0.5); Weight of time automatically 1-wgd)
```

# Example
To run the example, run

```
python3 ./Finalign_CLI.py
        —input_fasta ./example/H5Nx_2344b_recent_mafft_500.fasta
        —metadata_csv ./example/H5Nx_2344b_recent_mafft_500_RBS_meta.csv
        —date YMD
        —out_dir ./result/
        —out_name H5Nx_2344b_recent_mafft_500
        —resolution_strategy d
        —resolution_mode clean
        —proceed_trimming no
        —n_threshold 0.02
        —wgd 0.5
```

# Dependencies
To run FINALIGN, please create the Conda environment using the provided environment.yml file:
```
conda env create -f environment.yml
conda activate FN
```

This will install all necessary dependencies, including:

* **Core libraries**: numpy, pandas, scipy, matplotlib, etc.

* **Bioinformatics tools**: biopython, ete3, treeswift

* **Visualization**: seaborn, yellowbrick, logomaker

You can find the [environment.yml](environment.yml) file in this repository.

# Google Colab
You can run FINALGIN on the Google Golab https://colab.research.google.com/drive/1FfuJgz5Mj_3WV1-_rJCrghUhJrWgVbQh?usp=sharing


# Citation
A paper describing FINALGIN principles and application is

* [journal]
