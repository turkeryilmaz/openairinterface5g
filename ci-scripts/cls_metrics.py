import time
import datetime
import json
import os
import csv
import logging
import cls_containerize
import threading

def startcollection(mySSH, yaml_path, services_list):
    # Get the list of container names
    container_name_list = [cls_containerize.GetContainerName(mySSH, svc) for svc in services_list]
    
    # Iterate over the container names and create a thread for each one
    for container_name in container_name_list:
        monitor_thread = threading.Thread(target=collect_metrics, args=(mySSH, yaml_path, container_name))
        monitor_thread.daemon = True
        monitor_thread.start()

def collect_metrics(mySSH, yaml_path, container_name):
    if container_name.startswith("no such service"):
        logging.error("Container not found. Skipping docker stats collection")
        return
    command = f"docker inspect {container_name} > /dev/null"
    result = mySSH.run(command)
    if result.returncode != 0:
        logging.error(f"No such container: {container_name}. Skipping docker stats collection")
        return
    csv_file = f"{yaml_path.split('/')[-1]}_{container_name}"
    output_file = f"{os.getcwd()}/../cmake_targets/log/dockerstats/{csv_file}.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, mode='w', newline='') as file:
        fieldnames = ['timestamp', 'container_id', 'name', 'cpu_perc', 'mem_usage', 'mem_perc', 'net_io', 'block_io', 'pids']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if file.tell() == 0:
            writer.writeheader()
        while True:
            command = f"docker stats --no-stream --format '{{{{json .}}}}' {container_name}"
            result = mySSH.run(command,silent=True)
            if result.returncode != 0:
                logging.warning(f"Container {container_name} is not longer active.")
                break
            stats = parse_docker_stats(result.stdout.strip())
            for stat in stats:
                writer.writerow(stat)        
            time.sleep(1)
def parse_docker_stats(output):
    stats = []
    for line in output.strip().splitlines():
        data = json.loads(line)
        stats.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'container_id': data['ID'],
            'name': data['Name'],
            'cpu_perc': data['CPUPerc'],
            'mem_usage': data['MemUsage'],
            'mem_perc': data['MemPerc'],
            'net_io': data['NetIO'],
            'block_io': data['BlockIO'],
            'pids': data['PIDs']
        })
    return stats