# Installation
Follow the procedure below to install the python package
```
git clone https://github.com/linyc253/fq_coupling.git
cd fq_coupling
conda activate {your_env}
pip install -e .
```

# Usage
The input is the Q3D capacitance matrix (csv file, unit is **fF**). The signal net must be named as 

* Ground: `GND`
* Floating qubit: `{qubit_name}_L` (pad 1), `{qubit_name}_R` (pad 2),  `{qubit_name}_read` (readout line),  `{qubit_name}_xy` (xy line).
* Single-ended qubit (not tested yet): `{qubit_name}_I` (floating island), `{qubit_name}_read` (readout line),  `{qubit_name}_xy` (xy line).

where readout and xy are optional. All readout resonator frequencies are assumed to be 6.0 GHz.

For more details, please refer to `example.ipynb`.