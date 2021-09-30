#!/usr/bin/env python
import torch
from utils.dataset import SimpleIterDataset
import onnxruntime as ort
torch.manual_seed(42)

if __name__ == "__main__":

    device = 'cuda'
    
    # AK8 General (ONNX & PyTorch disagree)
    data_config_name = 'data/ak8_points_pf_sv_full.yaml'
    import networks.particle_net_pf_sv as network_module

    # AK8 MD (ONNX & PyTorch disagree)
    #data_config_name = 'data/ak8_points_pf_sv_mass_decorr.yaml'
    #import networks.particle_net_pf_sv as network_module

    # AK4 CHS (ONNX & PyTorch disagree)
    #data_config_name = 'data/AK4/ak4_points_pf_sv_CHS_eta4p7.yaml'
    #import networks.particle_net_ak4_pf_sv as network_module

    # AK8 Mass Reg. (ONNX & PyTorch agree)
    #data_config_name = 'data/ak8_points_pf_sv_mass_regression.yaml'
    #import networks.particle_net_pf_sv_mass_regression as network_module
    
    model_state_dict = data_config_name.replace('.yaml','.pt')
    jit_model_save = data_config_name.replace('.yaml','_gpu_jit.pt')
    onnx_model = data_config_name.replace('.yaml','.onnx')

    data_config = SimpleIterDataset([], data_config_name, for_training=False).config
    
    model, model_info = network_module.get_model(data_config, inference=True)
    model.to(device)
    model = torch.jit.script(model)
    
    model.load_state_dict(torch.load(model_state_dict))
    print(model)

    torch.jit.save(model, jit_model_save)
    
    pf_features_dims = len(data_config.input_dicts['pf_features'])
    sv_features_dims = len(data_config.input_dicts['sv_features'])
    n_pf = 100
    n_sv = 10
    pf_points = torch.rand((1, 2, n_pf)).to(device)
    pf_features = torch.rand((1, pf_features_dims, n_pf)).to(device)
    pf_mask = torch.ones((1, 1, n_pf)).to(device)
    sv_points = torch.rand((1, 2, n_sv)).to(device)
    sv_features = torch.rand((1, sv_features_dims, n_sv)).to(device)
    sv_mask = torch.ones((1, 1, n_sv)).to(device)
    
    model.eval()
    with torch.no_grad():
        out = model(pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask)
    print('PyTorch:', out)

    ort_session = ort.InferenceSession(onnx_model)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        
    # compute ONNX Runtime output prediction
    ort_inputs = {'pf_points': to_numpy(pf_points),
                  'pf_features': to_numpy(pf_features),
                  'pf_mask': to_numpy(pf_mask),
                  'sv_points': to_numpy(sv_points),
                  'sv_features': to_numpy(sv_features),
                  'sv_mask': to_numpy(sv_mask)}

    ort_out = ort_session.run(None, ort_inputs)
    print('ONNX:', ort_out)
    
    
