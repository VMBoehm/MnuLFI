# MnuLFI
Testing strengths and pitfalls of likelihood-free inference with neural networks on the MassiveNus peak count data set

## Installing the MnuLFI modules
Clone the repo then run
```
pip install --user -e .
```

Then to test the install and generate the data:
```
import mnulfi.data as Data
Data._make_data()
```
