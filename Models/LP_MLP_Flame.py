import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

import Objects.flame_pytorch.flame as flame
from argparse import Namespace


class Encoder(nn.Module):
    def __init__(self, input, features):
        super(Encoder, self).__init__()
        self.downs = nn.ModuleList()
        
        #Downsampling
        self.downs.append(nn.Linear(input, features[0]))
        self.downs.append(nn.ReLU())
        for i in range(len(features) - 1):
            self.downs.append(nn.Linear(features[i], features[i+1]))
            self.downs.append(nn.ReLU())

    def forward(self, x):
        x = x.flatten(start_dim=1)
        for down in self.downs:
            x = down(x)

        return x



class Model(nn.Module):
    def __init__(self, 
                 config={
                     'num_points': 100,
                     'in_channels': 1, 
                     'out_channels': 1, 
                     'features': [128, 64, 32], 
                     'shape_params': 100, 
                     'expression_params': 100}, 
                 state_dict=None, 
                 pretrain=False, 
                 device="cuda"):
        
        super(Model, self).__init__()
        
        flame_config = Namespace(**{
                'batch_size': 1, 
                'dynamic_landmark_embedding_path': './Objects/flame_pytorch/model/flame_dynamic_embedding.npy', 
                'expression_params': config['expression_params'], 
                'flame_model_path': './Objects/flame_pytorch/model/generic_model.pkl', 
                'num_worker': 4, 
                'optimize_eyeballpose': True, 
                'optimize_neckpose': True, 
                'pose_params': 6, 
                'ring_loss_weight': 1.0, 
                'ring_margin': 0.5, 
                'shape_params': config['shape_params'], 
                'static_landmark_embedding_path': './Objects/flame_pytorch/model/flame_static_embedding.pkl', 
                'use_3D_translation': True, 
                'use_face_contour': True
            })

        self._flame = flame.FLAME(flame_config)
        self._pose_params = torch.zeros(1, 6, device=device)

        self.output_image_size = config['output_image_size']

        self.encoder = Encoder(config['num_beams']*3, config['features'])



        self.expression_layer = nn.Linear(config['features'][-1], config['expression_params'])
        self.shape_layer = nn.Linear(config['features'][-1], config['shape_params'])


        if state_dict:
            self.load_from_dict(state_dict)

        if pretrain:
            self.encoder.requires_grad_ = False

    def get_statedict(self):
        return {"Encoder": self.encoder.state_dict(),
                "ExpressionLayer": self.expression_layer.state_dict(),
                "ShapeLayer": self.shape_layer.state_dict(),}

    def load_from_dict(self, dict):
        self.encoder.load_state_dict(dict["Encoder"])
        self.expression_layer.load_state_dict(dict["ExpressionLayer"])
        self.shape_layer.load_state_dict(dict["ShapeLayer"])

    def forward(self, x):
        x = self.encoder(x)
        shape_estimates = self.shape_layer(x)
        expression_estimates = self.expression_layer(x)
        vertices, _ = self._flame(shape_estimates, expression_estimates, self._pose_params)

        # We directly output the vertex positions here
        return vertices, shape_estimates*100.0, expression_estimates*100.0
    




def test():
    config={'in_channels': 1, 'out_channels': 1, 'num_beams': 50, 'features': [32, 64, 128], 'output_image_size': [512, 256], 'shape_params': 100, 'expression_params': 100}
    
    rand_input = torch.randn((1, 50, 3)).cuda()

    model = Model(config).cuda()
    vertices, shape_estimates, expression_estimates = model(rand_input)
    
    print(type(model).__name__)
    #print(shape_params.shape)
    #print(expression_params.shape)

    #print(Losses.CountingLoss(points.reshape(4, 2, -1), y))
    #print(CHM_loss.apply(points.reshape(4, 2, -1), y))

    # Seems to be working

if __name__ == "__main__":
    test()