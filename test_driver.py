import os
import csv
import json
import time

from test_parameter import *

os.environ["RAY_DEDUP_LOGS"] = "0" # Remove dedup log
os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

import ray
import dill
import hydra
import torch
import numpy as np

from test_worker import TestWorker
from diffusion_policy.workspace.base_workspace import BaseWorkspace


def run_test():
    print(f'Test devices: {DEVICE_LIST if USE_GPU else ["cpu"]}')
    checkpoint = os.path.join(run_path, 'checkpoints', checkpoint_name)
    output_dir = os.path.join(run_path, 'inference')
    csv_path = os.path.join(run_path, f'test_results_{time.strftime("%d_%m_%Y_%H_%M_%S")}.csv')

    ## Load Diffusion Model
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    # ship the policy to workers on CPU; each Runner moves its own copy to
    # its assigned device from DEVICE_LIST
    policy.to('cpu')
    policy.eval()
    
    meta_agents = [Runner.remote(i) for i in range(min(NUM_META_AGENT, NUM_TEST))]

    curr_test = 0
    completed_tests = 0

    dist_history = []
    success_dist_history = []
    collision_rate_history = []
    raw_invalid_action_rate_history = []
    fallback_rate_history = []
    fallback_failed_rate_history = []
    timeout_rate_history = []
    path_efficiency_history = []
    spl_history = []
    final_goal_distance_history = []
    smoothness_cost_history = []
    invalid_action_rate_history = []
    candidate_rejection_rate_history = []
    candidate_score_history = []
    inference_time_history = []
    successes = 0

    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        job_list.append(meta_agent.job.remote(policy, curr_test))
        curr_test += 1

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Episode Number',
            'Travel Distance',
            'Start Goal Distance',
            'Oracle Path Length',
            'Success',
            'Timeout Rate',
            'Collision Rate',
            'Raw Invalid Action Rate',
            'Fallback Rate',
            'Fallback Failed Rate',
            'Path Efficiency',
            'SPL',
            'Final Goal Distance',
            'Smoothness Cost',
            'Invalid Action Rate',
            'Candidate Rejection Rate',
            'Average Candidate Score',
            'Average Selected Candidate Score',
            'Inference Time (ms)',
            'Generated Candidates',
            'Rejected Candidates',
            'Ranking Breakdown',
            'Ranking Log',
        ])

        try:
            while completed_tests < curr_test:
                done_id, job_list = ray.wait(job_list)
                done_jobs = ray.get(done_id)

                for job in done_jobs:
                    completed_tests += 1
                    metrics, info = job

                    writer.writerow([
                        info['episode_number'],
                        metrics['travel_dist'],
                        metrics['start_goal_distance'],
                        metrics['oracle_path_length'],
                        metrics['success_rate'],
                        metrics['timeout_rate'],
                        metrics['collision_rate'],
                        metrics['raw_invalid_action_rate'],
                        metrics['fallback_rate'],
                        metrics['fallback_failed_rate'],
                        metrics['path_efficiency'],
                        metrics['spl'],
                        metrics['final_goal_distance'],
                        metrics['smoothness_cost'],
                        metrics['invalid_action_rate'],
                        metrics['candidate_rejection_rate'],
                        metrics['average_candidate_score'],
                        metrics['average_selected_candidate_score'],
                        metrics['inference_time_ms'],
                        metrics['generated_candidates'],
                        metrics['rejected_candidates'],
                        json.dumps(metrics['ranking_breakdown'], sort_keys=True),
                        json.dumps(metrics['ranking_log'], sort_keys=True),
                    ])

                    if metrics['success_rate']:
                        successes += 1
                        success_dist_history.append(metrics['travel_dist'])
                    dist_history.append(metrics['travel_dist'])
                    collision_rate_history.append(metrics['collision_rate'])
                    raw_invalid_action_rate_history.append(
                        metrics['raw_invalid_action_rate']
                    )
                    fallback_rate_history.append(metrics['fallback_rate'])
                    fallback_failed_rate_history.append(
                        metrics['fallback_failed_rate']
                    )
                    timeout_rate_history.append(metrics['timeout_rate'])
                    path_efficiency_history.append(metrics['path_efficiency'])
                    spl_history.append(metrics['spl'])
                    final_goal_distance_history.append(
                        metrics['final_goal_distance']
                    )
                    smoothness_cost_history.append(metrics['smoothness_cost'])
                    invalid_action_rate_history.append(
                        metrics['invalid_action_rate']
                    )
                    candidate_rejection_rate_history.append(
                        metrics['candidate_rejection_rate']
                    )
                    candidate_score_history.append(
                        metrics['average_candidate_score']
                    )
                    inference_time_history.append(metrics['inference_time_ms'])
                    
                if curr_test < NUM_TEST:
                    job_list.append(meta_agents[info['id']].job.remote(policy, curr_test))
                    curr_test += 1

            print('|#Total test:', NUM_TEST)
            print('|#Length of dist_history:', len(dist_history))
            print('|#Average length:', np.array(dist_history).mean())
            print('|#Length std:', np.array(dist_history).std())
            print('|#Successes:', successes)
            print('\n=== Test Navigation Metrics ===')
            print('| Experiment Mode:', EXPERIMENT_MODE)
            print('| Number of Samples:', 1 if EXPERIMENT_MODE == 'A' else NUM_ACTION_SAMPLES)
            print('| Collision Rate:', np.mean(collision_rate_history))
            print('| Raw Invalid Action Rate:', np.mean(raw_invalid_action_rate_history))
            print('| Fallback Rate:', np.mean(fallback_rate_history))
            print('| Fallback Failed Rate:', np.mean(fallback_failed_rate_history))
            print('| Success Rate:', successes / NUM_TEST)
            print('| Timeout Rate:', np.mean(timeout_rate_history))
            print('| Path Efficiency:', np.mean(path_efficiency_history))
            print('| SPL:', np.mean(spl_history))
            print('| Final Goal Distance:', np.mean(final_goal_distance_history))
            print('| Smoothness Cost:', np.mean(smoothness_cost_history))
            print('| Invalid Action Rate:', np.mean(invalid_action_rate_history))
            print('| Candidate Rejection Rate:', np.mean(candidate_rejection_rate_history))
            print('| Average Candidate Score:', np.mean(candidate_score_history))
            print('| Inference Time (ms):', np.mean(inference_time_history))
            print(
                'Note: raw invalid action rate measures the policy before '
                'fallback; fallback rate measures safety-layer corrections.'
            )
            print('|#Length of success_dist_history:', len(success_dist_history))
            if success_dist_history:
                print('|#Average Success length:', np.mean(success_dist_history))
                print('|#Success Length std:', np.std(success_dist_history))

        except KeyboardInterrupt:
            print("CTRL_C pressed. Killing remote workers")
            for a in meta_agents:
                ray.kill(a)

@ray.remote(num_cpus=1)
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        # round-robin workers across the available GPUs
        devices = DEVICE_LIST if USE_GPU else ['cpu']
        self.device = torch.device(devices[meta_agent_id % len(devices)])

    def do_job(self, episode_number, policy):
        policy = policy.to(self.device)
        policy.eval()
        worker = TestWorker(self.meta_agent_id, policy, episode_number, device=self.device, save_image=SAVE_GIFS)
        worker.run_episode()

        perf_metrics = worker.perf_metrics
        del worker
        return perf_metrics

    def job(self, policy, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))

        metrics = self.do_job(episode_number, policy)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return metrics, info

if __name__ == '__main__':
    ray.init()
    for i in range(NUM_RUN):
        run_test()
