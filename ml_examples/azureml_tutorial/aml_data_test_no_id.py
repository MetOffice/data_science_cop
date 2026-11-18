import sys

import xarray


input_path = sys.argv[1]
print(f'loading data from {input_path}')

ds = xarray.open_dataset(input_path, engine="h5netcdf")
print(ds)

