"""
"wombo-convo" (c) by Ignacio Slater M.
"wombo-convo" is licensed under a
Creative Commons Attribution 4.0 International License.
You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
"""
from torch import nn


class InceptionModule(nn.Module):
    """
    Convolutional neural network based inception module implementation.

    The module is defined as follows:
        -   Two convolutional layers are applied directly to the inputs; C_1 and C_2
        -   C_1's output is fed to a 3x3 convolutional layer named D_1
        -   C_2's output is fed to a 1x1 convolutional layer named D_2
        -   A max pooling layer applied to the outputs of every convolutional layer of dimensions
            1x1; D_3
        -   A 1x1 convolutional layer applied directly to the input

    The output of the layer is then defined as the concatenation of D_1, D_2, D_3 and D_4.
    """

    def __init__(self, in_channels, ch_3x3_reduce=96, ch_5x5_reduce=16, ch_3x3=128, ch_5x5=32,
                 ch_pool_proj=32, ch_1x1=64):
        """
        Creates a new inception module as a PyTorch convolutional network.

        Args:
            in_channels:
                the number of input channels
            ch_3x3_reduce:
                the number of output channels of layer C_1
            ch_5x5_reduce:
                the number of outputs channels of layer C_2
            ch_3x3:
                the number output of channels of the layer (D_1) that follows C_1
            ch_5x5:
                the number output of channels of the layer (D_2) that follows C_2
            ch_pool_proj:
                the number of output channels of the max pooling layer
            ch_1x1:
                the number of output channels of D_4
        """
        super(InceptionModule, self).__init__()
        # Acá inicializa todos los parámetros

    def forward(self, x):
        # Calcula la salida como un tensor con cantidad de canales de
        # salida dado por ch_3x3 + ch_5x5 + ch_pool_proj + ch_1x1
        pass

        return
        pass
