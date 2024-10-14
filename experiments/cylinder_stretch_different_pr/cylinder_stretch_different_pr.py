import os
import sys

proj_eps_list = ['-1', '0', '-0.5']
pr_list = ['0.1', '0.2', '0.3', '0.4', '0.45', '0.46', '0.47', '0.48', '0.49', '0.495', '0.499', '0.4999']
deform_scale_list = ['3.0']

mesh_name = 'cylinder'
experiment_name = 'figure_' + os.path.basename(__file__)[:-3]

for proj_eps in proj_eps_list:
  for pr in pr_list:
    for deform_scale in deform_scale_list:
      try:
        command = './example -p ' + proj_eps +  ' -n ' + mesh_name \
          + ' -l stretch_longest_axis -t ' + deform_scale \
          + ' --ym 1e8 --pr ' + pr \
          + ' --experiment_name ' + experiment_name \
          + ' --tr 0.01' 
        print(command)
        os.system(command)
      except:
        print('Error: ' + mesh_name)
