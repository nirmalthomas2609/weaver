#!/usr/bin/env python
# coding: utf-8

# In[13]:


import torch
import numpy as np
from utils.dataset import SimpleIterDataset
import os
import onnxruntime as ort
import torch.autograd.profiler as profiler


# In[14]:


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# In[15]:


def create_ort_inputs(inputs, input_names):
    return {name: to_numpy(inputs[index]) for (index, name) in enumerate(input_names)}


# In[16]:


def run_ort_inference(onnx_path, inputs, input_names):
    ort_session = ort.InferenceSession(onnx_path)
    runSessionParams = ort.RunOptions()
    runSessionParams.log_severity_level = 2
    ort_out = ort_session.run(None, create_ort_inputs(inputs, input_names), run_options = runSessionParams)
    return ort_out


# In[17]:


def run_pt_inference(model, inputs):
    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        output = model(*inputs)
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='cpu_time_total'))
    return output


# In[18]:


def export_onnx(model, inputs, dynamic_axes, input_names, output_names, onnx_path, opset_version = 15):
    torch.onnx.export(model, inputs, onnx_path, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, opset_version=opset_version)
    model_output = run_pt_inference(model, inputs)
    ort_output = run_ort_inference(onnx_path, inputs, input_names)
    print ("Model output - \n{}".format(model_output))
    print ("ORT output - \n{}".format(ort_output))


# In[19]:


def flatten(l):
    result = list()
    for subL in l:
        result.extend(subL)
    return result


# In[20]:


def create_model_inputs(model_type, num_jets = 4, num_particles_per_jet = 1, device = 'cpu', expectedIntDtype = 'int32'):
    def get_feature_dimension(feature_name, model_types_dims_mapper, default_dims_mapper, model_type):
        dimensions = None
        if feature_name in default_dims_mapper.keys():
            dimensions = default_dims_mapper[feature_name]
        else:
            dimensions = model_types_dims_mapper[model_type][feature_name]
        return dimensions
    
    model_type_dims_mapper = {'ak8_points_pf_sv_full': {'pf_features': 25, 'sv_features': 11}, 'ak8_points_pf_sv_mass_decorr': {'pf_features': 20, 'sv_features': 11}, 'ak4_points_pf_sv_CHS_eta4p7': {'pf_features': 20, 'sv_features': 11}, 'ak8_points_pf_sv_mass_regression': {'pf_features': 25, 'sv_features': 11}}
    default_dims_mapper = {'pf_points': 2, 'pf_mask': 1, 'sv_points': 2, 'sv_mask': 1}
    input_names = ['pf_points', 'pf_features', 'pf_mask', 'sv_points', 'sv_features', 'sv_mask']
    input_names += ['batch_shapes_{}'.format(item) for item in input_names]
    
    inputs = list()
    for input_name in input_names:
        if 'batch_shapes_' not in input_name:
            dimensions = get_feature_dimension(input_name, model_type_dims_mapper, default_dims_mapper, model_type)    
            inputs.append(torch.rand((num_jets * num_particles_per_jet * dimensions, ), dtype = torch.float).to(device))
        else:
            dimensions = get_feature_dimension(input_name.replace('batch_shapes_', ''), model_type_dims_mapper, default_dims_mapper, model_type)    
            datatype = torch.int32 if expectedIntDtype == 'int32' else torch.int64
            inputs.append(torch.tensor(flatten([[dimensions, num_particles_per_jet] for _ in range(num_jets)]), dtype = datatype).to(device))
    return tuple(inputs), input_names


# In[21]:


def get_model(model_type, device = 'cpu'):
    data_config_name = 'data/AK4/{}.yaml'.format(model_type)
    if model_type != 'ak4_points_pf_sv_CHS_eta4p7':
        data_config_name = 'data/{}.yaml'.format(model_type)
    import networks.particle_net_ak4_pf_sv as network_module1
    import networks.particle_net_pf_sv as network_module2
    import networks.particle_net_pf_sv_mass_regression as network_module3
    
    model_state_dict = data_config_name.replace('.yaml','.pt')
    jit_model_save = data_config_name.replace('.yaml','_ragged_gpu_jit.pt')
    onnx_model = data_config_name.replace('.yaml','.onnx')
    
    data_config = SimpleIterDataset([], data_config_name, for_training=False).config
    model, model_info = None, None
    
    network_module = None
    if model_type == 'ak8_points_pf_sv_full':
        network_module = network_module2
    elif model_type == 'ak8_points_pf_sv_mass_decorr':
        network_module = network_module2
    elif model_type == 'ak4_points_pf_sv_CHS_eta4p7':
        network_module = network_module1
    else:
        network_module = network_module3
    
    model, model_info = network_module.get_model(data_config, for_inference=True)
    model = torch.jit.script(model)
    model.load_state_dict(torch.load(model_state_dict, map_location=torch.device(device)))
    model.eval()
    
    return model, model_info


# In[22]:


def export_multiple_models(model_types):
    models = list() 
    for model_type in model_types:
        print ("Export model type - {}".format(model_type)) 
        model, model_info = get_model(model_type)
        model_inputs, input_names = create_model_inputs(model_type, num_jets = 9, num_particles_per_jet = 3)
        onnx_path = "exported_models/{}.onnx".format(model_type)
        export_onnx(model, model_inputs, model_info['dynamic_axes'], input_names, model_info['output_names'], onnx_path)
        models.append(model)
    return models


# In[23]:


def test_exported_models(models, model_types, configs = [{'num_jets': 1, 'num_particles_per_jet': 13}, {'num_jets': 1, 'num_particles_per_jet': 1}, {'num_jets': 17, 'num_particles_per_jet': 3}, {'num_jets': 16, 'num_particles_per_jet': 20}]):
    for index, model_type in enumerate(model_types):
        model = models[index]
        onnx_path = "exported_models/{}.onnx".format(model_type)
        print ("Model Type - {}\n".format(model_type))
        print ("------------")
        for config in configs:
            print ("Config - Num Jets = {} | Num particles per jet = {}".format(config['num_jets'], config['num_particles_per_jet']))
            test_model_inputs, test_model_input_names = create_model_inputs(model_type, **config)
            print ("PT inference - \n{}\n".format(run_pt_inference(model, test_model_inputs)))
            print ("ORT inference - \n{}\n".format(run_ort_inference(onnx_path, test_model_inputs, test_model_input_names)))


# In[24]:

import random
import sys
#Test PT model
model_type = 'ak4_points_pf_sv_CHS_eta4p7'
device = sys.argv[1]
for index in range(2, 25):
    print ("JET SIZE = ", index)
    jet_count = index
    jet_size = 10
    for _ in range(10):
        model_inputs, names = create_model_inputs(model_type, num_jets=jet_count, num_particles_per_jet=jet_size, device=device)
        run_pt_inference(get_model(model_type, device = device)[0], model_inputs)
# model_inputs, names = create_model_inputs(model_type)

