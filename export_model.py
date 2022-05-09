#!/usr/bin/env python
import torch
from utils.dataset import SimpleIterDataset
import onnxruntime as ort
torch.manual_seed(42)
import sys

if __name__ == "__main__":

    device = 'cuda'
    
    # AK8 General
    data_config_name = sys.argv[2]
    import networks.particle_net_pf_sv as network_module

    # AK8 MD
    #data_config_name = 'data/ak8_points_pf_sv_mass_decorr.yaml'
    #import networks.particle_net_pf_sv as network_module

    # AK4 CHS
    #data_config_name = 'data/AK4/ak4_points_pf_sv_CHS_eta4p7.yaml'
    #import networks.particle_net_ak4_pf_sv as network_module

    # AK8 Mass Reg.
    #data_config_name = 'data/ak8_points_pf_sv_mass_regression.yaml'
    #import networks.particle_net_pf_sv_mass_regression as network_module
    
    model_state_dict = data_config_name.replace('.yaml','.pt')
    jit_model_save = data_config_name.replace('.yaml','_ragged_gpu_jit.pt')
    onnx_model = data_config_name.replace('.yaml','.onnx')
    export_onnx = sys.argv[1]

    data_config = SimpleIterDataset([], data_config_name, for_training=False).config
    
    model, model_info = network_module.get_model(data_config, for_inference=True)
    model.to(device)
    model = torch.jit.script(model)
    
    model.load_state_dict(torch.load(model_state_dict))
    model.eval()

    print(model)
    inputs = tuple(
    torch.ones(model_info['input_shapes'][k], dtype=torch.float32) if ('batch_shapes_' not in k) else (torch.tensor([[len(data_config.input_dicts[k.replace('batch_shapes_', '')]), data_config.input_length[k.replace('batch_shapes_', '')]]], dtype=torch.int32)) for k in model_info['input_names'])
    torch.onnx.export(model, inputs, export_onnx,
                  input_names=model_info['input_names'],
                  output_names=model_info['output_names'],
                  dynamic_axes=model_info.get('dynamic_axes', None),
                  opset_version=13)