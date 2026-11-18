import datetime
import argparse
import pathlib
import subprocess


def get_cmd_args():
    parser = argparse.ArgumentParser(
        prog='aws download',
        description='small script to download met office ensemble data from public s3 (ASDI)',
    )
    process_time = lambda s1: datetime.datetime.strptime(s1, '%Y%m%dT%H%M')
    parser.add_argument('--vars', dest='var_list', type=str, nargs='*')
    parser.add_argument('--leadtimes',type=int,nargs='*')
    parser.add_argument('--start-init-time',dest='start_init_time',type=process_time)
    parser.add_argument('--num-init-times',dest='num_init_times',type=int)
    parser.add_argument('--src-path', dest='src_path', type=str)
    parser.add_argument('--dataset',type=str)
    parser.add_argument('--dest-dir', dest='dest_dir', type=pathlib.Path, default=pathlib.Path('.'))
    
    cmd_args = parser.parse_args()
    return cmd_args

def main():
    cmd_args = get_cmd_args()

    select_dataset = cmd_args.dataset
    root_s3_path = cmd_args.src_path
    ds_s3_path = root_s3_path + select_dataset

    root_out_dir = cmd_args.dest_dir
    dest_dir = root_out_dir / select_dataset
    if not dest_dir.is_dir():
        dest_dir.mkdir(parents=True)

    var_list = cmd_args.var_list
    print(var_list)

    lt_list = cmd_args.leadtimes

    


    s3_cmd_template = 'aws s3 cp --no-sign-request {src_path} {dest_path}'

    dt_template = '{dt.year:04d}{dt.month:02d}{dt.day:02d}T{dt.hour:02d}{dt.minute:02d}Z'
    init_dir_template = '/{fit.year:04d}/{fit.month:02d}/{fit.day:02d}/T{fit.hour:02d}{fit.minute:02d}Z
    lt_template = 'PT{lt:04d}H00M'
    fname_template = '{vt_str}-{lt_str}-{var}.nc'

    init_step = datetime.timedelta(hours=6)

    start_init = cmd_args.start_init_time
    num_inits=cmd_args.num_init_times
    
    for init_index in range(num_inits):
        forecast_init_time = start_init + (init_index * init_step)
        init_dir_path = ds_s3_path + init_dir_template.format(fit=forecast_init_time)
        print(init_dir_path)
        for var_name in var_list:
            print(var_name)
            for lt in lt_list:
                validity_time = forecast_init_time + datetime.timedelta(hours=lt)
                current_fname = fname_template.format(
                    vt_str=dt_template.format(dt=validity_time),
                    lt_str=lt_template.format(lt=lt),
                    var=var_name,               
                )
                src_path = init_dir_path + '/' + current_fname
                s3_cp_cmd = s3_cmd_template.format(
                    src_path=src_path,
                    dest_path = dest_dir,
                )
                print(f'copying file {src_path}')
                subprocess.call(
                    s3_cp_cmd,
                    shell=True,
                )
                # print(f'processed file {current_fname}')
            

if __name__ == "__main__":
    main()
    
