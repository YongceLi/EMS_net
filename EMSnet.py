import torch
import torch.nn as nn
import pdb

class Double_Conv_Block(nn.Module):
    """ Double_Conv_Block represents the (Conv x 2) Block in the local representation learning in the paper
    
    Attributes:
        curr_conv: torch sequential conv block as described in the paper.
    """

    def __init__(self, n_in, n_out, kernal_size, padding_size = 0):
        """ initialization
    
        Args:
            n_in: number of input channels
            n_out: number of filters, output channel size
            kernel_size: size for the kernel
            padding_size: size for the padding, default no padding
        """
        super(Double_Conv_Block, self).__init__()
        self.curr_conv = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size=kernal_size, padding = padding_size, stride=1),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(n_out, n_out, kernel_size=kernal_size, stride=1),
            nn.LeakyReLU(inplace = True)
        )
    
    def forward(self, input):
        """ make a forward pass """
        output = self.curr_conv(input)
        return output

    def __call__(self, input):
        """ make the class callable """
        return self.forward(input)
    
class Local_Representation(nn.Module):
    """ local representation learning without dense layer
    
    Attributes:
        curr_conv: the three sequential double conv block as described in the paper
    """
    def __init__(self):
        """ initialization """
        super(Local_Representation, self).__init__()
        self.curr_conv = nn.Sequential(
            Double_Conv_Block(1, 16, (1, 5), (0, 0)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
            nn.Dropout(p=0.5, inplace=False),
            Double_Conv_Block(16, 32, (1, 3), (0, 1)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
            nn.Dropout(p=0.5, inplace=False),
            Double_Conv_Block(32, 64, (1, 3), (0, 1)),
            nn.MaxPool2d(kernel_size=(1, 70), stride=2),
            nn.Dropout(p=0.5, inplace=False)
        )

    def forward(self, input):
        """ make a forward pass """
        output = self.curr_conv(input)
        return output

    def __call__(self, input):
        """ make the class callable """
        return self.forward(input)

class Global_Representation(nn.Module):
    """ Global Representation learning part
    
    Attributes:
        curr_conv: the sequential double conv as described in the paper (Part B in Fig. 2)
    """
    def __init__(self):
        """ initialization """
        super(Global_Representation, self).__init__()
        self.curr_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = (9, 300), stride = 1),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(16, 64, kernel_size = (1, 1), stride = 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        """ make a forward pass """
        output = self.curr_conv(input)
        return output

    def __call__(self, input):
        """ make the class callable """
        return self.forward(input)
    
class Feature_Combined_Dense(nn.Module):
    """ Dense layers in the combined feature learning part
    
    Attributes:
        curr_conv: Dense 1 to Logistic Sigmoid block (Part C Fig. 2)
    """
    def __init__(self):
        """ initialization """
        super(Feature_Combined_Dense, self).__init__()
        self.curr_conv = nn.Sequential(
            nn.Linear(4480, 512),
            nn.LeakyReLU(inplace = True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        """ make a forward pass """
        output = self.curr_conv(input)
        return output

    def __call__(self, input):
        """ make the class callable """
        return self.forward(input)

class EMS_Nets(nn.Module):
    """ The complete EMS_Nets described in Fig. 2 
    
    Attributes:
        Linear_local: The fully connected dense layer in Part(A) Fig. 2
        classifier_local: Sigmoid classifier in Part(A)
        local_representation_learning: the local representation learning part without dense layer in Part(A)
        global_representation_learning: the global representation learning part in Part(B)
        batch_size: the input batch size
        feature_combined_dense: the dense layer in Part(C)

    """
    def __init__(self, batch_size):
        super(EMS_Nets, self).__init__()
        self.Linear_local = nn.Sequential(
            nn.Linear(64, 64),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.classifier_local = nn.Sigmoid()
        self.local_representation_learning = Local_Representation()
        self.global_representation_learning = Global_Representation()
        self.batch_size = batch_size
        self.feature_combined_dense = Feature_Combined_Dense()

    def forward(self, input):
        """ make a forward pass 
        
        Args:
            input: input of size (N, H, W) N: batch_size, H: height, W: width
        """

        # Local Representation Learning
        # input shape Batch_size * 39 * 300
        local_outputs = []

        for i in range(input.shape[1]):
            curr_input = input[:, i, :]
            # choose every single channel of the input, shape: batchsize * 300
            # change the data to shape: batch_size * channel * h * w (batch_size * 1 * 1 * 300)
            curr_input = torch.unsqueeze(curr_input, dim = 1)
            curr_input = torch.unsqueeze(curr_input, dim = 1)
            
            curr_output = self.local_representation_learning(curr_input)
            # after the three double conv blocks, shape: batch_size * 64 * 1 * 1
            
            # change data to batch_size * 1 * 64 to pass in to the linear layer
            curr_output = torch.squeeze(curr_output, dim = 3)
            curr_output = torch.squeeze(curr_output, dim = 2)
            curr_output = torch.unsqueeze(curr_output, dim = 1) 

            # pass to dense layer
            curr_output = self.Linear_local(curr_output) 
            curr_output = self.classifier_local(curr_output)
            
            local_outputs.append(curr_output) 
            # every element is of shape: batch_size * 1 * 64
        
        local_representation = torch.cat(local_outputs, dim = 1)
        # batch_size * 39 * 64

        # Global representation learning
        global_input = torch.unsqueeze(input, dim = 1) # shape: Batch_size * 1 * 39 * 300 (N, C, H, W)
        global_output = self.global_representation_learning(global_input)
        # shape: N * 64 * 31 * 1
        
        global_output = torch.squeeze(global_output, dim = 3)
        global_representation = global_output.permute(0, 2, 1)
        # shape: batch_size * 31 * 64

        # Feature Combination
        combined_feature = torch.cat([local_representation, global_representation], dim = 1)
        
        # flatten feature
        flattened_combined_feature = combined_feature.view(input.shape[0], -1)
        output = self.feature_combined_dense(flattened_combined_feature)
        return output
    
    def __call__(self, input):
        """ make the class callable """
        return self.forward(input)




