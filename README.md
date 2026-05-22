# spspiral-cest
### Code for cardiac CEST (with spatial-spectral saturation) using VD spiral readouts in Pulseq

## Instructions
To use these sequences on your scanner (with and without spatial-spectral saturation pulses) follow the instructions below.

1.  Clone the repository to your machine using `git clone https://github.com/jweigandwhittier/spspiral-cest`
2.  Set up a [Conda](https://www.anaconda.com/docs/getting-started/miniconda/main) environment using the included `environment.yml` file
    * Navigate to your `spspiral-cest` directory
    * Run `conda env create -f environment.yml`
    * Activate the environment with `conda activate spspiral-cest`
3. Write a basic [WASABI](https://pubmed.ncbi.nlm.nih.gov/26857219/) sequence by running `write_wasabi.py`
    * You will be prompted for a patient heart rate, for non-cardiac scans set the `TRIGGER` flag to `False` and this will not be used
    * It is crucial that the resolution of your WASABI acquisition matches the resolution of your planned, corresponding CEST acquisition!
4. Optionally, run the WASABI sequence using your Pulseq interpreter of choice and reconstruct to obtain B0/B1 maps
5. Use the reconstructed B1 map to write a CEST sequence using `write_vdspiral_cest.py`
    * To write a sequence with equivalent Gaussian saturation pulses, set the `SPSP` flag to `False`
    * To write a full Z-spectral acquisition, set the `ZSPEC` flag to `True`
        * ⚠ All offsets are written to a single sequence, as such this is not suitable for cardiac scans with breath holds!

If you want to play with the spatial-spectral pulse code (or write a sequence with a generic SPSP pulse), use `spsp_example.py` or `cindy_example_b1.npy`

## Disclaimer
This code has, as of now, only been tested on Siemens hardware (GE forthcoming). Reconstruction code will NOT work directly with raw data from other platforms. 

These sequences are for research purposes only.

## References
If you use these sequences in your own work, please cite: 
* [Ayala C, Luo H, Godines K, et al. Individually tailored spatial–spectral pulsed CEST MRI for ratiometric mapping of myocardial energetic species at 3T. Magnetic Resonance in Med. 2023;90(6):2321-2333. doi:10.1002/mrm.29801](https://pubmed.ncbi.nlm.nih.gov/37526176/)

If you use WASABI in your own work, please cite the original paper: 
* [Schuenke P, Windschuh J, Roeloffs V, Ladd ME, Bachert P, Zaiss M. Simultaneous mapping of water shift and B 1 (WASABI)—Application to field‐Inhomogeneity correction of CEST MRI data. Magn Reson Med. 2017;77(2):571-580. doi:10.1002/mrm.26133](https://pubmed.ncbi.nlm.nih.gov/26857219/) 

Also consider referencing:
* [Layton KJ, Kroboth S, Jia F, et al. Pulseq: A rapid and hardware-independent pulse sequence prototyping framework: Rapid Hardware-Independent Pulse Sequence Prototyping. Magn Reson Med. 2017;77(4):1544-1552. doi:10.1002/mrm.26235](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.26235)
* [Herz K, Mueller S, Perlman O, et al. Pulseq‐CEST: Towards multi‐site multi‐vendor compatibility and reproducibility of CEST experiments using an open‐source sequence standard. Magnetic Resonance in Med. 2021;86(4):1845-1858. doi:10.1002/mrm.28825](https://pubmed.ncbi.nlm.nih.gov/33961312/)
* [Liebeskind A, Schüre JR, Fabian MS, et al. The Pulseq-CEST Library: definition of preparations and simulations, example data, and example evaluations. Magn Reson Mater Phy. 2025;38(3):413-422. doi:10.1007/s10334-025-01242-6](https://link.springer.com/article/10.1007/s10334-025-01242-6)
