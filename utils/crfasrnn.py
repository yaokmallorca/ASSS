import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class DenseCRFParams(object):
    """
    Parameters for the DenseCRF model
    """

    def __init__(
        self,
        alpha=160.0,
        beta=3.0,
        gamma=3.0,
        spatial_ker_weight=3.0,
        bilateral_ker_weight=5.0,
    ):
        """
        Default values were taken from https://github.com/sadeepj/crfasrnn_keras. More details about these parameters
        can be found in https://arxiv.org/pdf/1210.5644.pdf
        Args:
            alpha:                  Bandwidth for the spatial component of the bilateral filter
            beta:                   Bandwidth for the color component of the bilateral filter
            gamma:                  Bandwidth for the spatial filter
            spatial_ker_weight:     Spatial kernel weight
            bilateral_ker_weight:   Bilateral kernel weight
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.spatial_ker_weight = spatial_ker_weight
        self.bilateral_ker_weight = bilateral_ker_weight

class PermutoFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q_in, features):
        q_out = permuto_cpp.forward(q_in, features)[0]
        ctx.save_for_backward(features)
        return q_out

    @staticmethod
    def backward(ctx, grad_q_out):
        feature_saved = ctx.saved_tensors[0]
        grad_q_back = permuto_cpp.backward(
            grad_q_out.contiguous(), feature_saved.contiguous()
        )[0]
        return grad_q_back, None  # No need of grads w.r.t. features


def _spatial_features(image, sigma):
    """
    Return the spatial features as a Tensor
    Args:
        image:  Image as a Tensor of shape (channels, height, wight)
        sigma:  Bandwidth parameter
    Returns:
        Tensor of shape [h, w, 2] with spatial features
    """
    sigma = float(sigma)
    _, h, w = image.size()
    x = torch.arange(start=0, end=w, dtype=torch.float32, device=_CPU)
    xx = x.repeat([h, 1]) / sigma

    y = torch.arange(
        start=0, end=h, dtype=torch.float32, device=torch.device("cpu")
    ).view(-1, 1)
    yy = y.repeat([1, w]) / sigma

    return torch.stack([xx, yy], dim=2)


class AbstractFilter(ABC):
    """
    Super-class for permutohedral-based Gaussian filters
    """

    def __init__(self, image):
        self.features = self._calc_features(image)
        self.norm = self._calc_norm(image)

    def apply(self, input_):
        output = PermutoFunction.apply(input_, self.features)
        return output * self.norm

    @abstractmethod
    def _calc_features(self, image):
        pass

    def _calc_norm(self, image):
        _, h, w = image.size()
        all_ones = torch.ones((1, h, w), dtype=torch.float32, device=_CPU)
        norm = PermutoFunction.apply(all_ones, self.features)
        return 1.0 / (norm + _EPS)


class SpatialFilter(AbstractFilter):
    """
    Gaussian filter in the spatial ([x, y]) domain
    """

    def __init__(self, image, gamma):
        """
        Create new instance
        Args:
            image:  Image tensor of shape (3, height, width)
            gamma:  Standard deviation
        """
        self.gamma = gamma
        super(SpatialFilter, self).__init__(image)

    def _calc_features(self, image):
        return _spatial_features(image, self.gamma)


class BilateralFilter(AbstractFilter):
    """
    Gaussian filter in the bilateral ([r, g, b, x, y]) domain
    """

    def __init__(self, image, alpha, beta):
        """
        Create new instance
        Args:
            image:  Image tensor of shape (3, height, width)
            alpha:  Smoothness (spatial) sigma
            beta:   Appearance (color) sigma
        """
        self.alpha = alpha
        self.beta = beta
        super(BilateralFilter, self).__init__(image)

    def _calc_features(self, image):
        xy = _spatial_features(
            image, self.alpha
        )  # TODO Possible optimisation, was calculated in the spatial kernel
        rgb = (image / float(self.beta)).permute(1, 2, 0)  # Channel last order
        return torch.cat([xy, rgb], dim=2)

class CrfRnn(nn.Module):
    """
    PyTorch implementation of the CRF-RNN module described in the paper:
    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015 (https://arxiv.org/abs/1502.03240).
    """

    def __init__(self, num_labels, num_iterations=5, crf_init_params=None):
        """
        Create a new instance of the CRF-RNN layer.
        Args:
            num_labels:         Number of semantic labels in the dataset
            num_iterations:     Number of mean-field iterations to perform
            crf_init_params:    CRF initialization parameters
        """
        super(CrfRnn, self).__init__()

        if crf_init_params is None:
            crf_init_params = DenseCRFParams()

        self.params = crf_init_params
        self.num_iterations = num_iterations

        self._softmax = torch.nn.Softmax(dim=0)

        self.num_labels = num_labels

        # --------------------------------------------------------------------------------------------
        # --------------------------------- Trainable Parameters -------------------------------------
        # --------------------------------------------------------------------------------------------

        # Spatial kernel weights
        self.spatial_ker_weights = nn.Parameter(
            crf_init_params.spatial_ker_weight
            * torch.eye(num_labels, dtype=torch.float32)
        )

        # Bilateral kernel weights
        self.bilateral_ker_weights = nn.Parameter(
            crf_init_params.bilateral_ker_weight
            * torch.eye(num_labels, dtype=torch.float32)
        )

        # Compatibility transform matrix
        self.compatibility_matrix = nn.Parameter(
            torch.eye(num_labels, dtype=torch.float32)
        )

    def forward(self, image, logits):
        """
        Perform CRF inference.
        Args:
            image:  Tensor of shape (3, h, w) containing the RGB image
            logits: Tensor of shape (num_classes, h, w) containing the unary logits
        Returns:
            log-Q distributions (logits) after CRF inference
        """
        if logits.shape[0] != 1:
            raise ValueError("Only batch size 1 is currently supported!")

        image = image[0]
        logits = logits[0]

        spatial_filter = SpatialFilter(image, gamma=self.params.gamma)
        bilateral_filter = BilateralFilter(
            image, alpha=self.params.alpha, beta=self.params.beta
        )
        _, h, w = image.shape
        cur_logits = logits

        for _ in range(self.num_iterations):
            # Normalization
            q_values = self._softmax(cur_logits)

            # Spatial filtering
            spatial_out = torch.mm(
                self.spatial_ker_weights,
                spatial_filter.apply(q_values).view(self.num_labels, -1),
            )

            # Bilateral filtering
            bilateral_out = torch.mm(
                self.bilateral_ker_weights,
                bilateral_filter.apply(q_values).view(self.num_labels, -1),
            )

            # Compatibility transform
            msg_passing_out = (
                spatial_out + bilateral_out
            )  # Shape: (self.num_labels, -1)
            msg_passing_out = torch.mm(self.compatibility_matrix, msg_passing_out).view(
                self.num_labels, h, w
            )

            # Adding unary potentials
            cur_logits = msg_passing_out + logits

        return torch.unsqueeze(cur_logits, 0)