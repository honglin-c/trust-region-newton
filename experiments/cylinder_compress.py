import os

proj_eps_list = ['-1', '0', '-0.5']
pr_list = ['0.495', '0.35']
deform_scale_list = ['0.5']

mesh_name = 'cylinder_20'
experiment_name = 'figure_' + os.path.basename(__file__)[:-3]

for proj_eps in proj_eps_list:
  for pr in pr_list:
    for deform_scale in deform_scale_list:
      try:
        command = './example -p ' + proj_eps +  ' -n ' + mesh_name \
          + ' -l compress_longest_axis -t ' + deform_scale \
          + ' --ym 1e8 --pr ' + pr \
          + ' --experiment_name ' + experiment_name \
          + ' --tr 0.1' 
        print(command)
        os.system(command)
      except:
        print('Error: ' + mesh_name)
