# SCAOPO
It is the code of the paper "Successive Convex Approximation Based Off-Policy Optimization for Constrained Reinforcement Learning", which is authored by Chang Tian, An Liu, Guang Huang and Wu Luo. One can get the pdf document at https://arxiv.org/abs/2105.12545.

Run the file main_program.py to get results. In the main code, use example_name = 'MIMO_Gaussian' and example_name = 'MIMO_Beta' to respectively get the learning curves of the SCAOPO with the Gaussian policy and the Beta policy in the MIMO power allocation scenario. Use example_name = 'CLQR_Gaussian' and example_name = 'CLQR_Beta' to respectively get the results of the SCAOPO with the Gaussian policy and the Beta policy in the CLQR scenario.

There are some differences on the hyper-parameters in different examples. We have added comments in our code. One needs to do a few changes on main_program.py and modules.py when getting the results in different examples. How to do changes is instructed in the comments of the code.
