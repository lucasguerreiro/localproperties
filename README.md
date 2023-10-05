# localproperties

This repository explores local properties in complex networks. The main objective of this program is to identify how local properties provide realiable information of the entire network.
The main files are presented and discussed below. One need just to set them up and run in order to reproduce the results.

- gen_nets.py
	Functions to generate the network files.

- gen_walks.py
	Functions to perform walks on the nets and store the walks sequences.

- gen_props_parallel.py
	The functions in this file performs the walks in the nets and saves the resulting properties in separate files.

- join_props.py
	File that unifies the results obtained from the walks and joins them into a single file per setup

- gen_plots.py
	File containing the plot generations functions.

- util.py
	Helpful functions called by the other files and functions.

Authors:
Lucas Guerreiro;
Filipi Nascimento;
Diego Amancio.
