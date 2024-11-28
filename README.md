# RTFAST Spectra

This project is the revised version of the RTFAST-spectra (Ricketts et al 2025). This version of the emulator takes a 
distinctly different strategy from the previous work: emulation of purely the reflection spectrum. This allows us to reduce
the complexity of the emulated spectrum as well as keeping as much of the original model analytical as possible. We also
expand to emulate the entire spectrum from 0.1-100keV. This allows for the use of instruments such as NuSTAR, which are key
in the fitting of the high energy rollover.

As such, this incorporates much of the machine learning structure developed in the previous work for a slightly different 
emulated spectrum.

## Installation
Currently, installation is restricted to active use of this repository. This repository is not built for generic training
and is restricted to its particular use case. Feel free to fork this repository for your own project, citing credit for
the framework to this repository.

## Usage
Used to train neural network emulators for the RTFAST-spectra project.

## Support
Please feel free to open an issue on this git repository or reach out to b.j.ricketts(at)uva.nl.

## Contributing
If you would like to contribute to this project, please reach out to b.j.ricketts(at)uva.nl.

## Authors and acknowledgment

## License
This project operates under the MIT open license (?).

## Project status
Active development
