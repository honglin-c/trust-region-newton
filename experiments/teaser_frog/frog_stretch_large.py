import os

proj_eps_list = ['-0.5', '-1', '0']
pr_list = ['0.495', '0.3']

deform_scale = '-1.5'
deform_label = 'stretch_shear_front2'

mesh_name = 'frog'
experiment_name = 'figure_' + os.path.basename(__file__)[:-3]

for pr in pr_list:
  for proj_eps in proj_eps_list:
    try:
      command = './example -p ' + proj_eps +  ' -n ' + mesh_name \
        + ' -l ' + deform_label + ' -g ' + deform_scale \
        + ' --ym 1e8 --pr ' + pr \
        + ' --experiment_name ' + experiment_name \
        + ' --tr 0.01' \
        + ' -b 0.2'
      print(command)
      os.system(command)
    except:
      print('Error: ' + mesh_name)