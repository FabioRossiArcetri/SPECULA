import os
import torch
import torch.nn as nn
from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue

class Conv2dNetTester(BaseProcessingObj):
    def __init__(self,
                 network_filename,
                 nmodes = 5,
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.verbose = True
        self.network_filename = network_filename
        self.nmodes = nmodes

        self.device = torch.device("cpu")
        
        if os.path.isfile(self.network_filename):
            self.model = torch.load(self.network_filename, map_location=self.device)
            self.model.eval()
            if self.verbose:
                print(f"[{self.name}] Loaded model from {self.network_filename}")
        else:
            raise FileNotFoundError(f"Model file not found at {self.network_filename}")
        
        self.loss_fn = nn.MSELoss()
        self.inputs['input_2d_batch'] = InputValue(type=BaseValue)
        self.inputs['labels'] = InputValue(type=BaseValue)
        self.outputs["loss"] = None
        self.outputs["prediction"] = None

        self.meanp = 0
        self.stdp = 884.0701904296875
        self.meanmodes = 0 # -10.492571141174995
        self.stdmodes = 382.37760440241357

    def rescale(self, v):        
        mean = self.xp.mean(v)
        v -= mean
        std = self.xp.std(v)
        v /= std
        return v, mean, std

    def trigger(self):
        x_in = self.local_inputs['input_2d_batch']
        y_in = self.local_inputs['labels']

        if x_in is None or y_in is None:
            return

        ph = x_in.get_value()[:, 1] * x_in.get_value()[:, 0]
        print(f"{ph.shape}")        
        modes = y_in.get_value()[:, 1:self.nmodes+1]
        print(f"{modes.shape}")

        ph = ph[:, self.xp.newaxis, :, :]  # shape (B, 1, H, H)

        print(f"[{self.name}]  modes: {modes}")

#        ph, self.meanp, self.stdp = self.rescale(ph)
#        modes, self.meanmodes, self.stdmodes = self.rescale(modes)

        ph -= self.meanp
        ph /= self.stdp
        modes -= self.meanmodes
        modes /= self.stdmodes

        inputs = torch.tensor(ph, dtype=torch.float32, device=self.device)
        targets = torch.tensor(modes, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            preds = self.model(inputs)
            loss = self.loss_fn(preds, targets).item()

        self.outputs["loss"] = loss
        self.outputs["prediction"] = preds
        if self.verbose:
            print(f"[{self.name}] Eval loss: {loss:.6f}, preds: {preds}, targets: {targets}")

    def post_trigger(self):
        super().post_trigger()
