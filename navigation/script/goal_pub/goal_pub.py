import os
import logging
import sys
import socket
import numpy as np
import rospy
import torch
import torch.nn as nn
from mpi4py import MPI

from goal_pub_config import PUBGOAL


def run(comm, env):
    while not rospy.is_shutdown():
        env.pub_goal_func()


if __name__ == '__main__':

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()
    
    env = PUBGOAL(index=rank)
    
    print("START")

    try:
        run(comm=comm, env=env)
    except KeyboardInterrupt:
        pass
