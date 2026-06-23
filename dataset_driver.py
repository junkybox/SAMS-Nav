import os
import csv
import time

from dataset_parameter import *

os.environ["RAY_DEDUP_LOGS"] = "0" # Remove dedup log

import ray
import zarr
import torch
import numpy as np

from dataset_worker import DatasetWorker


def append_navigation_episode(zarr_path, episode_data):
    zarr_group = zarr.open_group(zarr_path, mode='a')
    data_keys = (
        'node_inputs',
        'node_padding_mask',
        'edge_mask',
        'current_index',
        'current_edge',
        'edge_padding_mask',
        'action',
        'goal',
        'pose',
    )

    if 'data' not in zarr_group:
        data_group = zarr_group.create_group('data')
        for key in data_keys:
            data_group.create_dataset(
                key, data=np.asarray(episode_data[key]), chunks=True
            )
    else:
        data_group = zarr_group['data']
        for key in data_keys:
            data_group[key].append(np.asarray(episode_data[key]))

    episode_end = np.asarray(episode_data['episode_end'], dtype=np.int32)
    if 'meta' not in zarr_group:
        meta_group = zarr_group.create_group('meta')
        meta_group.create_dataset('episode_ends', data=episode_end, chunks=True)
    else:
        meta_group = zarr_group['meta']
        meta_group['episode_ends'].append(
            episode_end + meta_group['episode_ends'][-1]
        )


def run_test():
    device = torch.device(DEVICE if USE_GPU else 'cpu')
    print(f'Dataset device: {device}')
    os.makedirs(dataset_path, exist_ok=True)
    csv_path = (
        f'{dataset_path}/navigation_results_'
        f'{time.strftime("%d_%m_%Y_%H_%M_%S")}.csv'
    )

    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]

    next_episode = 0
    completed_attempts = 0
    successful_episodes = 0
    num_failed = 0

    dist_history = []

    job_list = []
    for meta_agent in meta_agents[:min(NUM_META_AGENT, NUM_EPISODES)]:
        job_list.append(meta_agent.job.remote(next_episode))
        next_episode += 1

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Episode Number',
            'Travel Distance',
            'Start Goal Distance',
            'Oracle Path Length',
            'Expert Path Length',
            'Path Efficiency',
            'SPL',
            'Path Steps',
            'Success',
            'Expert Planner Map',
        ])

        try:
            while successful_episodes < NUM_EPISODES:
                done_id, job_list = ray.wait(job_list)
                done_jobs = ray.get(done_id)

                for job in done_jobs:
                    completed_attempts += 1
                    episode_data, metrics, info = job

                    writer.writerow([
                        info["episode_number"],
                        metrics['travel_dist'],
                        metrics['start_goal_distance'],
                        metrics['oracle_path_length'],
                        metrics['expert_path_length'],
                        metrics['path_efficiency'],
                        metrics['spl'],
                        metrics['path_steps'],
                        metrics['success_rate'],
                        metrics['expert_planner_map'],
                    ])
                    
                    if metrics['success_rate'] != True:
                        print(f'Episode {info["episode_number"]} failed')
                        num_failed += 1
                    else:
                        dist_history.append(metrics['travel_dist'])
                        zarr_path = f'{dataset_path}/data.zarr'
                        append_navigation_episode(zarr_path, episode_data)
                        successful_episodes += 1

                    if successful_episodes + len(job_list) < NUM_EPISODES:
                        job_list.append(
                            meta_agents[info['id']].job.remote(next_episode)
                        )
                        next_episode += 1

            print('|#Saved navigation episodes:', successful_episodes)
            print('|#Total collection attempts:', completed_attempts)
            if dist_history:
                print('|#Average path length:', np.mean(dist_history))
                print('|#Path length std:', np.std(dist_history))
            print('|#Failed:', num_failed)

        except KeyboardInterrupt:
            print("CTRL_C pressed. Killing remote workers")
            for a in meta_agents:
                ray.kill(a)


@ray.remote(num_cpus=1)
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device(DEVICE if USE_GPU else 'cpu')
        
    def do_job(self, episode_number):
        worker = DatasetWorker(self.meta_agent_id, episode_number, device=self.device, save_image=SAVE_GIFS, greedy=True)
        worker.run_episode()

        episode_data = worker.episode_data
        perf_metrics = worker.perf_metrics
        return episode_data, perf_metrics

    def job(self, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))

        episode_data, metrics = self.do_job(episode_number)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return episode_data,metrics, info


if __name__ == '__main__':
    ray.init()
    for i in range(NUM_RUN):
        run_test()
