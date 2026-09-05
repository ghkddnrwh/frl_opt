# export_wandb_final_160_runs.py
#
# Final hardcoded W&B exporter for the complete experiment grid:
#
#   4 MuJoCo environments
#     - Ant
#     - HalfCheetah
#     - Hopper
#     - Walker2d
#
#   2 perturbation types
#     - gravity
#     - friction
#
#   4 algorithm variants
#     - PPO Avg
#     - AMPO Uniform
#     - AMPO Adaptive, dual_lr=1e-4
#     - AMPO Adaptive, dual_lr=3e-4
#
#   5 seeds
#     - 1, 2, 3, 4, 5
#
# Total = 4 * 2 * 4 * 5 = 160 unique finished runs.
#
# CSV files are NOT needed at runtime. All 160 run IDs are hardcoded.
#
# Usage:
#   pip install wandb
#   wandb login
#   python export_wandb_final_160_runs.py

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import wandb


DEFAULT_ENTITY = "ukjo19"
DEFAULT_PROJECT = "sb3"
DEFAULT_OUT_ROOT = "logs/wandb_logs_final_160"
DEFAULT_PAGE_SIZE = 500
DEFAULT_RETRIES = 3


RUNS = [{'id': 'ia5g9obf',
  'variant': 'ppo_avg',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': '1t9soclb',
  'variant': 'ppo_avg',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'qx6uiwo1',
  'variant': 'ppo_avg',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'x6yx9l73',
  'variant': 'ppo_avg',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': '214b9ldm',
  'variant': 'ppo_avg',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'q60duchj',
  'variant': 'ppo_avg',
  'env': 'halfcheetah',
  'perturbation': 'gravity',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'eejhzina',
  'variant': 'ppo_avg',
  'env': 'halfcheetah',
  'perturbation': 'gravity',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'km8fqamu',
  'variant': 'ppo_avg',
  'env': 'halfcheetah',
  'perturbation': 'gravity',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'm43o04j1',
  'variant': 'ppo_avg',
  'env': 'halfcheetah',
  'perturbation': 'gravity',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'q57b0mu0',
  'variant': 'ppo_avg',
  'env': 'halfcheetah',
  'perturbation': 'gravity',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': '57klq9o5',
  'variant': 'ppo_avg',
  'env': 'hopper',
  'perturbation': 'gravity',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'az0fxslz',
  'variant': 'ppo_avg',
  'env': 'hopper',
  'perturbation': 'gravity',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'nneeci7q',
  'variant': 'ppo_avg',
  'env': 'hopper',
  'perturbation': 'gravity',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': '9h88qj5t',
  'variant': 'ppo_avg',
  'env': 'hopper',
  'perturbation': 'gravity',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'k2x7o90t',
  'variant': 'ppo_avg',
  'env': 'hopper',
  'perturbation': 'gravity',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'jz02rc8j',
  'variant': 'ppo_avg',
  'env': 'walker2d',
  'perturbation': 'gravity',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': '6ko0jjfx',
  'variant': 'ppo_avg',
  'env': 'walker2d',
  'perturbation': 'gravity',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'x4mc01tm',
  'variant': 'ppo_avg',
  'env': 'walker2d',
  'perturbation': 'gravity',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'bi4gryxr',
  'variant': 'ppo_avg',
  'env': 'walker2d',
  'perturbation': 'gravity',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': '5x88t2y0',
  'variant': 'ppo_avg',
  'env': 'walker2d',
  'perturbation': 'gravity',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'i577xg7r',
  'variant': 'ppo_avg',
  'env': 'walker2d',
  'perturbation': 'friction',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'roc6pu0n',
  'variant': 'ppo_avg',
  'env': 'walker2d',
  'perturbation': 'friction',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': '1tg2w61z',
  'variant': 'ppo_avg',
  'env': 'walker2d',
  'perturbation': 'friction',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'ut3wpzmk',
  'variant': 'ppo_avg',
  'env': 'walker2d',
  'perturbation': 'friction',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'x3phbmno',
  'variant': 'ppo_avg',
  'env': 'walker2d',
  'perturbation': 'friction',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'jhfalusc',
  'variant': 'ppo_avg',
  'env': 'hopper',
  'perturbation': 'friction',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'bgc2zipd',
  'variant': 'ppo_avg',
  'env': 'hopper',
  'perturbation': 'friction',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': '2vs7cj2o',
  'variant': 'ppo_avg',
  'env': 'hopper',
  'perturbation': 'friction',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': '4nr2t8zm',
  'variant': 'ppo_avg',
  'env': 'hopper',
  'perturbation': 'friction',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': '5be4r7rl',
  'variant': 'ppo_avg',
  'env': 'hopper',
  'perturbation': 'friction',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'b4ycegsm',
  'variant': 'ppo_avg',
  'env': 'halfcheetah',
  'perturbation': 'friction',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': '1htevbt3',
  'variant': 'ppo_avg',
  'env': 'halfcheetah',
  'perturbation': 'friction',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'azhvv3v2',
  'variant': 'ppo_avg',
  'env': 'halfcheetah',
  'perturbation': 'friction',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': '9lzb745o',
  'variant': 'ppo_avg',
  'env': 'halfcheetah',
  'perturbation': 'friction',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'f8ldrlar',
  'variant': 'ppo_avg',
  'env': 'halfcheetah',
  'perturbation': 'friction',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'uuo4ptd7',
  'variant': 'ppo_avg',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'iom5ybgn',
  'variant': 'ppo_avg',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'fstu8qun',
  'variant': 'ppo_avg',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': '7oqn6cjo',
  'variant': 'ppo_avg',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'w5lv7uvu',
  'variant': 'ppo_avg',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': '5gk6mlu2',
  'variant': 'ampo_uniform',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': '8btm4ezm',
  'variant': 'ampo_uniform',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'v09m2pbi',
  'variant': 'ampo_uniform',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'bf3imy2m',
  'variant': 'ampo_uniform',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'q5gxe6r7',
  'variant': 'ampo_uniform',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'p4iuycv6',
  'variant': 'ampo_uniform',
  'env': 'halfcheetah',
  'perturbation': 'friction',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'zqu8a8cl',
  'variant': 'ampo_uniform',
  'env': 'halfcheetah',
  'perturbation': 'friction',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': '0y5ghyu7',
  'variant': 'ampo_uniform',
  'env': 'halfcheetah',
  'perturbation': 'friction',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'x3mj8w9r',
  'variant': 'ampo_uniform',
  'env': 'halfcheetah',
  'perturbation': 'friction',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'smfg99ll',
  'variant': 'ampo_uniform',
  'env': 'halfcheetah',
  'perturbation': 'friction',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': '2e8o5l8q',
  'variant': 'ampo_uniform',
  'env': 'hopper',
  'perturbation': 'friction',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'vq9gymfg',
  'variant': 'ampo_uniform',
  'env': 'hopper',
  'perturbation': 'friction',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': '5xf9exfr',
  'variant': 'ampo_uniform',
  'env': 'hopper',
  'perturbation': 'friction',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': '535nxqnd',
  'variant': 'ampo_uniform',
  'env': 'hopper',
  'perturbation': 'friction',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'pxj33jw2',
  'variant': 'ampo_uniform',
  'env': 'hopper',
  'perturbation': 'friction',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'axtfjrx7',
  'variant': 'ampo_uniform',
  'env': 'walker2d',
  'perturbation': 'friction',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'lmq5kiuv',
  'variant': 'ampo_uniform',
  'env': 'walker2d',
  'perturbation': 'friction',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'hb30ng3h',
  'variant': 'ampo_uniform',
  'env': 'walker2d',
  'perturbation': 'friction',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'd4oesiiv',
  'variant': 'ampo_uniform',
  'env': 'walker2d',
  'perturbation': 'friction',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'qrjjyxzn',
  'variant': 'ampo_uniform',
  'env': 'walker2d',
  'perturbation': 'friction',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'klbyaafg',
  'variant': 'ampo_uniform',
  'env': 'walker2d',
  'perturbation': 'gravity',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'ngopjb4i',
  'variant': 'ampo_uniform',
  'env': 'walker2d',
  'perturbation': 'gravity',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'ur1to35d',
  'variant': 'ampo_uniform',
  'env': 'walker2d',
  'perturbation': 'gravity',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'kzhmgvqe',
  'variant': 'ampo_uniform',
  'env': 'walker2d',
  'perturbation': 'gravity',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'zetrz8df',
  'variant': 'ampo_uniform',
  'env': 'walker2d',
  'perturbation': 'gravity',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': '5y9zp1xr',
  'variant': 'ampo_uniform',
  'env': 'hopper',
  'perturbation': 'gravity',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'em4tgr50',
  'variant': 'ampo_uniform',
  'env': 'hopper',
  'perturbation': 'gravity',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'lrqjv341',
  'variant': 'ampo_uniform',
  'env': 'hopper',
  'perturbation': 'gravity',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'bpb47sm4',
  'variant': 'ampo_uniform',
  'env': 'hopper',
  'perturbation': 'gravity',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'kkod87jr',
  'variant': 'ampo_uniform',
  'env': 'hopper',
  'perturbation': 'gravity',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'myemsjql',
  'variant': 'ampo_uniform',
  'env': 'halfcheetah',
  'perturbation': 'gravity',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': '27tzdqkx',
  'variant': 'ampo_uniform',
  'env': 'halfcheetah',
  'perturbation': 'gravity',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'x84fcrg8',
  'variant': 'ampo_uniform',
  'env': 'halfcheetah',
  'perturbation': 'gravity',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'srqobu27',
  'variant': 'ampo_uniform',
  'env': 'halfcheetah',
  'perturbation': 'gravity',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'kokra7dm',
  'variant': 'ampo_uniform',
  'env': 'halfcheetah',
  'perturbation': 'gravity',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'qc1aistw',
  'variant': 'ampo_uniform',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'yn952l4n',
  'variant': 'ampo_uniform',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'iuibv3kn',
  'variant': 'ampo_uniform',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'vf9rfobi',
  'variant': 'ampo_uniform',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'qysddk52',
  'variant': 'ampo_uniform',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': None},
 {'id': 'fj778472',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': '02rzx00q',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'jlf4ki0a',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'h3dxbntu',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'p3enpd7s',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'fcz7wzak',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'qqtkbux0',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'sk3tddj9',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'y2wf93lo',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'c3vc2x2u',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'ant',
  'perturbation': 'friction',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': '257omi2u',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'halfcheetah',
  'perturbation': 'friction',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'bmzieulj',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'halfcheetah',
  'perturbation': 'friction',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'p08x6vb4',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'halfcheetah',
  'perturbation': 'friction',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': '0pl7ie7l',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'halfcheetah',
  'perturbation': 'friction',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': '1mie8qn7',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'halfcheetah',
  'perturbation': 'friction',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'sij2zuht',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'halfcheetah',
  'perturbation': 'friction',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'h4smdalz',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'halfcheetah',
  'perturbation': 'friction',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'cusaxdbj',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'halfcheetah',
  'perturbation': 'friction',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'kw5cdkkw',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'halfcheetah',
  'perturbation': 'friction',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'sv6n20xp',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'halfcheetah',
  'perturbation': 'friction',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 't316lqh6',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'hopper',
  'perturbation': 'friction',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'z8yvc5wi',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'hopper',
  'perturbation': 'friction',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': '2mmdf5cw',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'hopper',
  'perturbation': 'friction',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'vuiywo3y',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'hopper',
  'perturbation': 'friction',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 't6n0jo3g',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'hopper',
  'perturbation': 'friction',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': '2sg943z9',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'hopper',
  'perturbation': 'friction',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'c85nehdg',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'hopper',
  'perturbation': 'friction',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'd85w589k',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'hopper',
  'perturbation': 'friction',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'ofh7y2hl',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'hopper',
  'perturbation': 'friction',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': '53xbedou',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'hopper',
  'perturbation': 'friction',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'fxba195i',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'walker2d',
  'perturbation': 'friction',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'ww28q347',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'walker2d',
  'perturbation': 'friction',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'gqdspk3g',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'walker2d',
  'perturbation': 'friction',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'zh1lcu91',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'walker2d',
  'perturbation': 'friction',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': '05b5a3gh',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'walker2d',
  'perturbation': 'friction',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'aj3s9tlu',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'walker2d',
  'perturbation': 'friction',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'nqgoxgwf',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'walker2d',
  'perturbation': 'friction',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'd6jm7gu3',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'walker2d',
  'perturbation': 'friction',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'dlfxco38',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'walker2d',
  'perturbation': 'friction',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'weoa5i1v',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'walker2d',
  'perturbation': 'friction',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'ym38zlx6',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'walker2d',
  'perturbation': 'gravity',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': '6wipal9p',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'walker2d',
  'perturbation': 'gravity',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': '9nwkmz4h',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'walker2d',
  'perturbation': 'gravity',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': '92gd8e1x',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'walker2d',
  'perturbation': 'gravity',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'x25pmp07',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'walker2d',
  'perturbation': 'gravity',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': '2tgnotr7',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'walker2d',
  'perturbation': 'gravity',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': '2gbsxvb9',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'walker2d',
  'perturbation': 'gravity',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': '9u3rv1o2',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'walker2d',
  'perturbation': 'gravity',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'jgf88dem',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'walker2d',
  'perturbation': 'gravity',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'fpmjysbj',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'walker2d',
  'perturbation': 'gravity',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'ldjd1mue',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'hopper',
  'perturbation': 'gravity',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': '1tr3dnqy',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'hopper',
  'perturbation': 'gravity',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'fhkf7uhe',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'hopper',
  'perturbation': 'gravity',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'xasogzen',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'hopper',
  'perturbation': 'gravity',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'k3bqd2oz',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'hopper',
  'perturbation': 'gravity',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'vopan76j',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'hopper',
  'perturbation': 'gravity',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': '08ebpbod',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'hopper',
  'perturbation': 'gravity',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'buyif834',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'hopper',
  'perturbation': 'gravity',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': '2j1xtq0a',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'hopper',
  'perturbation': 'gravity',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'zfhft0yn',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'hopper',
  'perturbation': 'gravity',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'dnrn2b1t',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'halfcheetah',
  'perturbation': 'gravity',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'k0va83ir',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'halfcheetah',
  'perturbation': 'gravity',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': '9nexx9r6',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'halfcheetah',
  'perturbation': 'gravity',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'q3si00b9',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'halfcheetah',
  'perturbation': 'gravity',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'xa8boisn',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'halfcheetah',
  'perturbation': 'gravity',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'suq43xd4',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'halfcheetah',
  'perturbation': 'gravity',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'tbb2wlaf',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'halfcheetah',
  'perturbation': 'gravity',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'hnr7d64e',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'halfcheetah',
  'perturbation': 'gravity',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': '763gi9ua',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'halfcheetah',
  'perturbation': 'gravity',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'ki77184e',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'halfcheetah',
  'perturbation': 'gravity',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'qo7i6fll',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 's5cqw7pq',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'a8a1u0m6',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'scktpqrq',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': 'xk4ftv2f',
  'variant': 'ampo_adaptive_dual_lr_3e-4',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': 0.0003},
 {'id': '5ufm7ndd',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 5,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'ignh5lfe',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 4,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': '5o829tv0',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 3,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 'ngpnketj',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 2,
  'csv_state': 'finished',
  'dual_lr': 0.0001},
 {'id': 's9ux3zid',
  'variant': 'ampo_adaptive_dual_lr_1e-4',
  'env': 'ant',
  'perturbation': 'gravity',
  'seed': 1,
  'csv_state': 'finished',
  'dual_lr': 0.0001}]


def validate_hardcoded_grid():
    assert len(RUNS) == 160, f"Expected 160 runs, got {len(RUNS)}"

    ids = [r["id"] for r in RUNS]
    assert len(set(ids)) == 160, "Duplicate run IDs found"

    variants = [
        "ppo_avg",
        "ampo_uniform",
        "ampo_adaptive_dual_lr_1e-4",
        "ampo_adaptive_dual_lr_3e-4",
    ]
    envs = ["ant", "halfcheetah", "hopper", "walker2d"]
    perturbations = ["gravity", "friction"]

    for variant in variants:
        for env in envs:
            for perturbation in perturbations:
                seeds = sorted(
                    r["seed"]
                    for r in RUNS
                    if r["variant"] == variant
                    and r["env"] == env
                    and r["perturbation"] == perturbation
                )
                assert seeds == [1, 2, 3, 4, 5], (
                    variant,
                    env,
                    perturbation,
                    seeds,
                )


def json_safe(x):
    if x is None:
        return None

    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else x

    if isinstance(x, (str, int, bool)):
        return x

    if isinstance(x, datetime):
        return x.isoformat()

    try:
        import numpy as np

        if isinstance(x, np.integer):
            return int(x)

        if isinstance(x, np.floating):
            x = float(x)
            return None if math.isnan(x) or math.isinf(x) else x

        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass

    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]

    return str(x)


def dump_json(path: Path, obj):
    path.write_text(
        json.dumps(json_safe(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_output_dir(out_root: Path, spec: dict) -> Path:
    return (
        out_root
        / spec["variant"]
        / spec["env"]
        / spec["perturbation"]
        / f"seed_{spec['seed']}__{spec['id']}"
    )


def export_one_run(
    api,
    spec: dict,
    entity: str,
    project: str,
    out_root: Path,
    page_size: int,
    retries: int,
    skip_existing: bool,
    only_finished: bool,
):
    run_id = spec["id"]
    api_path = f"{entity}/{project}/{run_id}"
    out = run_output_dir(out_root, spec)
    success_marker = out / "_SUCCESS.json"

    if skip_existing and success_marker.exists():
        print("    SKIP: already exported")
        return {
            "status": "already_exported",
            "id": run_id,
            **spec,
            "out_dir": str(out),
        }

    out.mkdir(parents=True, exist_ok=True)

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            run = api.run(api_path)
            state = str(getattr(run, "state", "")).lower()

            if only_finished and state != "finished":
                return {
                    "status": "skipped_not_finished",
                    "id": run_id,
                    "api_state": state,
                    **spec,
                }

            config = {
                k: v
                for k, v in dict(run.config).items()
                if not str(k).startswith("_")
            }
            dump_json(out / "config.json", config)

            try:
                summary = dict(run.summary)
            except Exception:
                try:
                    summary = run.summary._json_dict
                except Exception:
                    summary = {}
            dump_json(out / "summary.json", summary)

            metadata = {
                "hardcoded_spec": spec,
                "api_path": api_path,
                "entity": getattr(run, "entity", None),
                "project": getattr(run, "project", None),
                "id": getattr(run, "id", None),
                "name": getattr(run, "name", None),
                "state": getattr(run, "state", None),
                "created_at": getattr(run, "created_at", None),
                "url": getattr(run, "url", None),
                "tags": getattr(run, "tags", None),
                "notes": getattr(run, "notes", None),
            }
            dump_json(out / "metadata.json", metadata)

            history_path = out / "history.jsonl"
            n_rows = 0
            all_keys = set()
            first_rows = []
            last_rows = []

            # Rewrite history from scratch for this attempt.
            with history_path.open("w", encoding="utf-8") as f:
                for row in run.scan_history(page_size=page_size):
                    row = json_safe(row)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

                    n_rows += 1
                    all_keys.update(row.keys())

                    if len(first_rows) < 5:
                        first_rows.append(row)

                    last_rows.append(row)
                    if len(last_rows) > 5:
                        last_rows.pop(0)

            readme = [
                "# W&B Run Export",
                "",
                f"- Variant: `{spec['variant']}`",
                f"- Environment: `{spec['env']}`",
                f"- Perturbation: `{spec['perturbation']}`",
                f"- Seed: `{spec['seed']}`",
                f"- Dual LR: `{spec.get('dual_lr')}`",
                f"- API path: `{api_path}`",
                f"- Run name: `{getattr(run, 'name', None)}`",
                f"- Run id: `{run_id}`",
                f"- State at export: `{state}`",
                f"- History rows: `{n_rows}`",
                "",
                "## Files",
                "- `metadata.json`: run metadata + hardcoded experiment spec",
                "- `config.json`: W&B config",
                "- `summary.json`: W&B summary",
                "- `history.jsonl`: full scalar history from scan_history()",
                "- `_SUCCESS.json`: successful completion marker",
                "",
                "## Logged keys",
                ", ".join(f"`{k}`" for k in sorted(all_keys)),
                "",
                "## Summary",
                "```json",
                json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
                "```",
                "",
                "## Config",
                "```json",
                json.dumps(json_safe(config), ensure_ascii=False, indent=2),
                "```",
                "",
                "## First 5 history rows",
                "```json",
                json.dumps(first_rows, ensure_ascii=False, indent=2),
                "```",
                "",
                "## Last 5 history rows",
                "```json",
                json.dumps(last_rows, ensure_ascii=False, indent=2),
                "```",
            ]

            (out / "README_for_GPT.md").write_text(
                "\n".join(readme),
                encoding="utf-8",
            )

            result = {
                "status": "exported",
                "id": run_id,
                "variant": spec["variant"],
                "env": spec["env"],
                "perturbation": spec["perturbation"],
                "seed": spec["seed"],
                "dual_lr": spec.get("dual_lr"),
                "api_state": state,
                "history_rows": n_rows,
                "out_dir": str(out),
            }

            dump_json(success_marker, result)
            return result

        except Exception as e:
            last_error = e
            print(
                f"    attempt {attempt}/{retries} failed: "
                f"{type(e).__name__}: {e}"
            )
            if attempt < retries:
                time.sleep(min(2 ** attempt, 10))

    return {
        "status": "error",
        "id": run_id,
        "variant": spec["variant"],
        "env": spec["env"],
        "perturbation": spec["perturbation"],
        "seed": spec["seed"],
        "dual_lr": spec.get("dual_lr"),
        "error": f"{type(last_error).__name__}: {last_error}",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export the final 160 hardcoded W&B runs."
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-export runs even when _SUCCESS.json already exists.",
    )
    parser.add_argument(
        "--include-not-finished",
        action="store_true",
        help="Export even if the current W&B API state is not finished.",
    )
    args = parser.parse_args()

    validate_hardcoded_grid()

    api = wandb.Api()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Save the full hardcoded manifest before starting.
    dump_json(out_root / "_hardcoded_160_run_manifest.json", RUNS)

    print("=" * 78)
    print("FINAL W&B EXPORT: 160 runs")
    print("  4 envs x 2 perturbations x 4 variants x 5 seeds")
    print(f"  entity/project : {args.entity}/{args.project}")
    print(f"  output         : {out_root}")
    print(f"  resume         : {not args.force}")
    print("=" * 78)
    print()

    results = []

    # Stable order for easier progress tracking.
    ordered_runs = sorted(
        RUNS,
        key=lambda r: (
            r["variant"],
            r["env"],
            r["perturbation"],
            r["seed"],
        ),
    )

    for i, spec in enumerate(ordered_runs, start=1):
        print(
            f"[{i:03d}/160] "
            f"{spec['variant']} / {spec['env']} / "
            f"{spec['perturbation']} / seed={spec['seed']} / "
            f"id={spec['id']}"
        )

        result = export_one_run(
            api=api,
            spec=spec,
            entity=args.entity,
            project=args.project,
            out_root=out_root,
            page_size=args.page_size,
            retries=max(1, args.retries),
            skip_existing=not args.force,
            only_finished=not args.include_not_finished,
        )
        results.append(result)

        if result["status"] == "exported":
            print(f"    DONE: {result['history_rows']} history rows")
        elif result["status"] == "already_exported":
            pass
        elif result["status"] == "skipped_not_finished":
            print(
                f"    SKIP: API state={result.get('api_state')} "
                "(use --include-not-finished to override)"
            )
        else:
            print(f"    ERROR: {result.get('error')}")

    counts = {}
    for item in results:
        counts[item["status"]] = counts.get(item["status"], 0) + 1

    per_variant = {}
    for item in results:
        variant = item["variant"]
        per_variant.setdefault(variant, {})
        status = item["status"]
        per_variant[variant][status] = (
            per_variant[variant].get(status, 0) + 1
        )

    final_summary = {
        "expected_total": 160,
        "status_counts": counts,
        "per_variant": per_variant,
        "results": results,
    }
    dump_json(out_root / "_export_summary.json", final_summary)

    print()
    print("=" * 78)
    print("EXPORT SUMMARY")
    for status, count in sorted(counts.items()):
        print(f"  {status:24s} {count}")
    print(f"  output: {out_root}")
    print("=" * 78)


if __name__ == "__main__":
    main()
