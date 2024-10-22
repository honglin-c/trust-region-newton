import os

proj_eps_list = ['-1', '0', '-0.5']
pr_list = ['0.3']

mesh_name = 'hand_closed'
experiment_name = 'figure_' + os.path.basename(__file__)[:-3]

for proj_eps in proj_eps_list:
  for pr in pr_list:
    try:
      command = './example -p ' + proj_eps +  ' -n ' + mesh_name \
        + ' -l bend_stretch -t -0.02' \
        + ' --ym 1e8 --pr ' + pr \
        + ' --experiment_name ' + experiment_name \
        + ' --tr 0.1' \
        + ' --rotate_ratio 0.15' 
      print(command)
      os.system(command)
    except:
      print('Error: ' + mesh_name)
