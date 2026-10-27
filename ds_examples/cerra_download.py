import cdsapi
import pathlib

dataset = "reanalysis-cerra-pressure-levels"
current_year = 2010
var_list = [
        "v_component_of_wind",
        "geopotential",
        "relative_humidity",
        "temperature",
        "u_component_of_wind",
    ]

request = {
    "data_type": ["reanalysis"],
    "product_type": ["analysis"],
    "variable": var_list,
    "year": [f'{current_year:04d}'],
    "month": ["01"],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00",
        "03:00",
        "06:00",
        "09:00",
        "12:00",
        "15:00",
        "18:00",
        "21:00",
    ],
    "pressure_level": [
        "200", "500", "750",
        "800", "1000"
    ],
    "data_format": "netcdf",

}

client = cdsapi.Client()

data_dir = pathlib.Path('/data/users/stephen.haddad/cerra/')

for var_name in var_list:
    for current_month in range(1,13):
        print(f'processing data for {var_name}, time {current_year:04d}-{current_month:02d}')
        fname = f'cerra_{var_name}_{current_year:04d}{current_month:02d}.nc'
        request['variable'] = [var_name]
        request['month'] = [f'{current_month:02d}']
        req1 = client.retrieve(dataset,
                        request,
                        str(data_dir / fname),
                       )
