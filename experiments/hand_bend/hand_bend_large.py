import os

proj_eps_list = ['-1', '0', '-0.5']
pr_list = ['0.48']
pr_list.reverse()

mesh_name = 'hand_closed'
experiment_name = 'figure_' + os.path.basename(__file__)[:-3]

for proj_eps in proj_eps_list:
  for pr in pr_list:
    try:
      command = './example -p ' + proj_eps +  ' -n ' + mesh_name \
        + ' -l bend_stretch -t 0.1' \
        + ' --ym 1e8 --pr ' + pr \
        + ' --experiment_name ' + experiment_name \
        + ' --tr 0.01' \
        + ' -c 4e-7' \
        + ' --rotate_ratio 0.5' 
      print(command)
      os.system(command)
    except:
      print('Error: ' + mesh_name)
