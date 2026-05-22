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
