## Installing the MnuLFI modules
set `$MNULFI_DIR` in your bashrc or bash_profile then run 
```
python setup.py install --user 
```

Then to test the install and generate the data: 
```
import mnulfi.data as Data
Data._make_data() 
```
