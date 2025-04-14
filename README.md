# FINALGIN
Multiple sequence alignment (MSA) analysis of viral genomes is indispensable for investigating mutations and phylogenetic relationships. Although many alignment tools, such as MAFFT, are widely used for aligning viral sequences, their results often require additional refinement to ensure codon-level accuracy, particularly in the presence of ambiguous bases and erroneous gaps. To address this, FINALIGN was developed as a post-alignment refinement tool to resolve ambiguities and correct misalignments in viral MSAs.

# Abstract Figure

![Figure1](https://github.com/user-attachments/assets/0ae0edb2-7b5a-49dd-afbc-bc674bece8e8)

# Quick start
Download FINALGIN and make sure [dependencies](#dependencies-and-os) are installed. For the quickest start, just run our example:

`[path/to/FINALIGN]/./FINALIGN -s`

You can retrieve example data in `data/example`.

# Command line
You can run FINALGIN from the command line as follows:

Run 'finalign -h' to display all available options. 
To run the example, run

```
python3 ./Finalign_CLI.py —input_fasta ../seq/H5Nx_2344b_recent_mafft_500.fasta —metadata_csv ../seq/H5Nx_2344b_recent_mafft_500_RBS_meta.csv —date YMD —out_dir ../result/ —out_name H5Nx_2344b_recent_mafft_500 —resolution_strategy d —resolution_mode clean —proceed_trimming no —n_threshold 0.02 —wgd 0.5
```

# Google Colab
You can run FINALGIN on the Google Golab https://colab.research.google.com/drive/1FfuJgz5Mj_3WV1-_rJCrghUhJrWgVbQh?usp=sharing


# Citation
A paper describing FINALGIN principles and application is

* [journal]
