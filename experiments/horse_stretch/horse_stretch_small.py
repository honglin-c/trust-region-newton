import os
import sys

proj_eps_list = ['-1', '0', '-0.5']
pr_list = ['0.3']
deform_scale_list = ['0.2']

mesh_name = 'horse'
experiment_name = 'figure_' + os.path.basename(__file__)[:-3]

for proj_eps in proj_eps_list:
  for pr in pr_list:
    for deform_scale in deform_scale_list:
      try:
        command = './example -p ' + proj_eps +  ' -n ' + mesh_name \
          + ' -l stretch_front -g ' + deform_scale \
          + ' --ym 1e8 --pr ' + pr \
          + ' --experiment_name ' + experiment_name 
        print(command)
        os.system(command)
      except:
        print('Error: ' + mesh_name)
