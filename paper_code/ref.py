param_names = ['propensity', 'walk_radius', 'arm_bias', 'arm_weight', 'atrisk_gathering_rate', 'mix_perc']
param_names_readable = ['Propensity', 'Walk Distance', 'Arm Bias', 'Arm Weight', 'At-Risk Gathering Rate', 'Mix Percent']
# param_units = ['Agent / Person / Year', 'Miles', 'Location / Mile^2', 'Unitless', '']
param_label_map = {k:v for k,v in zip(param_names, param_names_readable)}