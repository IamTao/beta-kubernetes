# -*- coding: utf-8 -*-
import os
import re
import argparse
import subprocess


def get_args():
    # feed them to the parser.
    parser = argparse.ArgumentParser(description='Extract results for plots')

    # add arguments.
    parser.add_argument(
        '--delete_dmlb_cluster', '-t', action='store_true', default=False)

    # parse args.
    args = parser.parse_args()
    return args


def write_txt(data, out_path, type="w"):
    """write the data to the txt file."""
    with open(out_path, type) as f:
        f.write(data)


def run_cmd(args):
    return subprocess.check_output(args).decode('utf-8').strip().split('\n')


def get_existing_pod_names():
    lines = run_cmd(['kubectl', 'get', 'pods'])[1:]
    existing_pods_info = [re.split(r'\s+', line) for line in lines]
    existing_pods_name = [
        l[0] for l in existing_pods_info
        if 'dmlb' in l[0] and 'Running' in l[2]]
    return existing_pods_name


def get_existing_pod_info(existing_pod_name):
    def get(pattern, lines):
        got_items = [re.findall(pattern, line, re.DOTALL) for line in lines]
        return [item for item in got_items if len(item) != 0][0][0]

    info = {}
    raw = run_cmd(['kubectl', 'describe', 'pod', existing_pod_name])
    info['name'] = get(r'^Name:\s+([\w-]+)', raw)
    print('    processing {}'.format(info['name']))

    info['namespace'] = get(r'^Namespace:\s+([\w-]+)', raw)
    info['ip'] = get(r'^IP:\s+([\d.]+)', raw)
    info['ports'] = get(r'Ports:\s+([\d\w.,/\s]+)', raw).split(', ')
    info['num_gpu'] = get(r'nvidia.com/gpu:\s+(\d)', raw)
    return info


def get_existing_pods_info(existing_pod_names):
    all_info = {}
    for existing_pod_name in existing_pod_names:
        all_info[existing_pod_name] = get_existing_pod_info(existing_pod_name)
    return all_info


def save_hostfile(all_info):
    ips = '\n'.join(
        ['{} slots={}'.format(info['ip'], info['num_gpu'])
         for key, info in all_info.items()]
    )
    write_txt(ips, 'kubernetes/hostfile')
    return ips


def setup_ssh(existing_pod_names):
    for existing_pod_name in existing_pod_names:
        print('setup ssh connection for {}...'.format(existing_pod_name))
        os.system(
            "kubectl exec -it {} bash 'entrypoint.sh'".format(
                existing_pod_name))


def main(args):
    print(' get pod names.')
    existing_pod_names = get_existing_pod_names()

    if len(existing_pod_names) == 0:
        print('  does not exist pods.')
        return

    print(' get pod info.')
    existing_pods_info = get_existing_pods_info(existing_pod_names)

    print(' get IPs and save them to path.')
    save_hostfile(existing_pods_info)

    print('setup ssh connections')
    setup_ssh(existing_pod_names)


if __name__ == '__main__':
    args = get_args()
    main(args)
