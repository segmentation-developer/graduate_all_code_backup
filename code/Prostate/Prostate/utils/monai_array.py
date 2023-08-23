
import warnings
from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor
from monai.networks import one_hot
from monai.networks.layers import GaussianFilter, apply_filter
from monai.transforms.transform import Transform
from monai.transforms.utils import fill_holes, get_largest_connected_component_mask, get_unique_labels
from monai.transforms.utils_pytorch_numpy_unification import unravel_index
from monai.utils import TransformBackends, convert_data_type, deprecated_arg, ensure_tuple, look_up_option
from monai.utils.type_conversion import convert_to_dst_type

__all__ = [
    "Activations",
    "AsDiscrete",
    "FillHoles",
    "KeepLargestConnectedComponent",
    "LabelFilter",
    "LabelToContour",
    "MeanEnsemble",
    "ProbNMS",
    "VoteEnsemble",
]


class Activations(Transform):
    """
    Add activation operations to the denseUnet_3D output, typically `Sigmoid` or `Softmax`.

    Args:
        sigmoid: whether to execute sigmoid function on denseUnet_3D output before transform.
            Defaults to ``False``.
        softmax: whether to execute softmax function on denseUnet_3D output before transform.
            Defaults to ``False``.
        other: callable function to execute other activation layers, for example:
            `other = lambda x: torch.tanh(x)`. Defaults to ``None``.

    Raises:
        TypeError: When ``other`` is not an ``Optional[Callable]``.

    """

    backend = [TransformBackends.TORCH]

    def __init__(self, sigmoid: bool = False, softmax: bool = False, other: Optional[Callable] = None) -> None:
        self.sigmoid = sigmoid
        self.softmax = softmax
        if other is not None and not callable(other):
            raise TypeError(f"other must be None or callable but is {type(other).__name__}.")
        self.other = other

    def __call__(
        self,
        img: NdarrayOrTensor,
        sigmoid: Optional[bool] = None,
        softmax: Optional[bool] = None,
        other: Optional[Callable] = None,
    ) -> NdarrayOrTensor:
        """
        Args:
            sigmoid: whether to execute sigmoid function on denseUnet_3D output before transform.
                Defaults to ``self.sigmoid``.
            softmax: whether to execute softmax function on denseUnet_3D output before transform.
                Defaults to ``self.softmax``.
            other: callable function to execute other activation layers, for example:
                `other = torch.tanh`. Defaults to ``self.other``.

        Raises:
            ValueError: When ``sigmoid=True`` and ``softmax=True``. Incompatible values.
            TypeError: When ``other`` is not an ``Optional[Callable]``.
            ValueError: When ``self.other=None`` and ``other=None``. Incompatible values.

        """
        if sigmoid and softmax:
            raise ValueError("Incompatible values: sigmoid=True and softmax=True.")
        if other is not None and not callable(other):
            raise TypeError(f"other must be None or callable but is {type(other).__name__}.")

        # convert to float as activation must operate on float tensor
        img_t, *_ = convert_data_type(img, torch.Tensor, dtype=torch.float)
        if sigmoid or self.sigmoid:
            img_t = torch.sigmoid(img_t)
        if softmax or self.softmax:
            img_t = torch.softmax(img_t, dim=0)

        act_func = self.other if other is None else other
        if act_func is not None:
            img_t = act_func(img_t)
        out, *_ = convert_to_dst_type(img_t, img)
        return out


class AsDiscrete(Transform):
    """
    Execute after denseUnet_3D forward to transform denseUnet_3D output to discrete values.
    It can complete below operations:

        -  execute `argmax` for input logits values.
        -  threshold input value to 0.0 or 1.0.
        -  convert input value to One-Hot format.
        -  round the value to the closest integer.

    Args:
        argmax: whether to execute argmax function on input data before transform.
            Defaults to ``False``.
        to_onehot: if not None, convert input data into the one-hot format with specified number of classes.
            Defaults to ``None``.
        threshold: if not None, threshold the float values to int number 0 or 1 with specified theashold.
            Defaults to ``None``.
        rounding: if not None, round the data according to the specified option,
            available options: ["torchrounding"].

    Example:

        >>> transform = AsDiscrete(argmax=True)
        >>> print(transform(np.array([[[0.0, 1.0]], [[2.0, 3.0]]])))
        # [[[1.0, 1.0]]]

        >>> transform = AsDiscrete(threshold=0.6)
        >>> print(transform(np.array([[[0.0, 0.5], [0.8, 3.0]]])))
        # [[[0.0, 0.0], [1.0, 1.0]]]

        >>> transform = AsDiscrete(argmax=True, to_onehot=2, threshold=0.5)
        >>> print(transform(np.array([[[0.0, 1.0]], [[2.0, 3.0]]])))
        # [[[0.0, 0.0]], [[1.0, 1.0]]]

    .. deprecated:: 0.6.0
        ``n_classes`` is deprecated, use ``to_onehot`` instead.

    .. deprecated:: 0.7.0
        ``num_classes`` is deprecated, use ``to_onehot`` instead.
        ``logit_thresh`` is deprecated, use ``threshold`` instead.
        ``threshold_values`` is deprecated, use ``threshold`` instead.

    """

    backend = [TransformBackends.TORCH]

    @deprecated_arg(name="n_classes", new_name="num_classes", since="0.6", msg_suffix="please use `to_onehot` instead.")
    @deprecated_arg("num_classes", since="0.7", msg_suffix="please use `to_onehot` instead.")
    @deprecated_arg("logit_thresh", since="0.7", msg_suffix="please use `threshold` instead.")
    @deprecated_arg(
        name="threshold_values", new_name="threshold", since="0.7", msg_suffix="please use `threshold` instead."
    )
    def __init__(
        self,
        argmax: bool = False,
        to_onehot: Optional[int] = None,
        threshold: Optional[float] = None,
        rounding: Optional[str] = None,
        n_classes: Optional[int] = None,  # deprecated
        num_classes: Optional[int] = None,  # deprecated
        logit_thresh: float = 0.5,  # deprecated
        threshold_values: Optional[bool] = False,  # deprecated
    ) -> None:
        self.argmax = argmax
        if isinstance(to_onehot, bool):  # for backward compatibility
            warnings.warn("`to_onehot=True/False` is deprecated, please use `to_onehot=num_classes` instead.")
            to_onehot = num_classes if to_onehot else None
        self.to_onehot = to_onehot

        if isinstance(threshold, bool):  # for backward compatibility
            warnings.warn("`threshold_values=True/False` is deprecated, please use `threshold=value` instead.")
            threshold = logit_thresh if threshold else None
        self.threshold = threshold

        self.rounding = rounding

    @deprecated_arg(name="n_classes", new_name="num_classes", since="0.6", msg_suffix="please use `to_onehot` instead.")
    @deprecated_arg("num_classes", since="0.7", msg_suffix="please use `to_onehot` instead.")
    @deprecated_arg("logit_thresh", since="0.7", msg_suffix="please use `threshold` instead.")
    @deprecated_arg(
        name="threshold_values", new_name="threshold", since="0.7", msg_suffix="please use `threshold` instead."
    )
    def __call__(
        self,
        img: NdarrayOrTensor,
        argmax: Optional[bool] = None,
        to_onehot: Optional[int] = None,
        threshold: Optional[float] = None,
        rounding: Optional[str] = None,
        n_classes: Optional[int] = None,  # deprecated
        num_classes: Optional[int] = None,  # deprecated
        logit_thresh: Optional[float] = None,  # deprecated
        threshold_values: Optional[bool] = None,  # deprecated
    ) -> NdarrayOrTensor:
        """
        Args:
            img: the input tensor data to convert, if no channel dimension when converting to `One-Hot`,
                will automatically add it.
            argmax: whether to execute argmax function on input data before transform.
                Defaults to ``self.argmax``.
            to_onehot: if not None, convert input data into the one-hot format with specified number of classes.
                Defaults to ``self.to_onehot``.
            threshold: if not None, threshold the float values to int number 0 or 1 with specified threshold value.
                Defaults to ``self.threshold``.
            rounding: if not None, round the data according to the specified option,
                available options: ["torchrounding"].

        .. deprecated:: 0.6.0
            ``n_classes`` is deprecated, use ``to_onehot`` instead.

        .. deprecated:: 0.7.0
            ``num_classes`` is deprecated, use ``to_onehot`` instead.
            ``logit_thresh`` is deprecated, use ``threshold`` instead.
            ``threshold_values`` is deprecated, use ``threshold`` instead.

        """
        if isinstance(to_onehot, bool):
            warnings.warn("`to_onehot=True/False` is deprecated, please use `to_onehot=num_classes` instead.")
            to_onehot = num_classes if to_onehot else None
        if isinstance(threshold, bool):
            warnings.warn("`threshold_values=True/False` is deprecated, please use `threshold=value` instead.")
            threshold = logit_thresh if threshold else None

        img_t, *_ = convert_data_type(img, torch.Tensor)
        if argmax or self.argmax:
            img_t = torch.argmax(img_t, dim=0, keepdim=True)

        to_onehot = self.to_onehot if to_onehot is None else to_onehot
        if to_onehot is not None:
            if not isinstance(to_onehot, int):
                raise AssertionError("the number of classes for One-Hot must be an integer.")
            img_t = one_hot(img_t, num_classes=to_onehot, dim=0)

        threshold = self.threshold if threshold is None else threshold
        if threshold is not None:
            img_t = img_t >= threshold

        rounding = self.rounding if rounding is None else rounding
        if rounding is not None:
            look_up_option(rounding, ["torchrounding"])
            img_t = torch.round(img_t)

        img, *_ = convert_to_dst_type(img_t, img, dtype=torch.float)
        return img


class KeepLargestConnectedComponent(Transform):
    """
    Keeps only the largest connected component in the image.
    This transform can be used as a post-processing step to clean up over-segment areas in denseUnet_3D output.

    The input is assumed to be a channel-first PyTorch Tensor:
      1) For not OneHot format data, the values correspond to expected labels,
      0 will be treated as background and the over-segment pixels will be set to 0.
      2) For OneHot format data, the values should be 0, 1 on each labels,
      the over-segment pixels will be set to 0 in its channel.

    For example:
    Use with applied_labels=[1], is_onehot=False, connectivity=1::

       [1, 0, 0]         [0, 0, 0]
       [0, 1, 1]    =>   [0, 1 ,1]
       [0, 1, 1]         [0, 1, 1]

    Use with applied_labels=[1, 2], is_onehot=False, independent=False, connectivity=1::

      [0, 0, 1, 0 ,0]           [0, 0, 1, 0 ,0]
      [0, 2, 1, 1 ,1]           [0, 2, 1, 1 ,1]
      [1, 2, 1, 0 ,0]    =>     [1, 2, 1, 0 ,0]
      [1, 2, 0, 1 ,0]           [1, 2, 0, 0 ,0]
      [2, 2, 0, 0 ,2]           [2, 2, 0, 0 ,0]

    Use with applied_labels=[1, 2], is_onehot=False, independent=True, connectivity=1::

      [0, 0, 1, 0 ,0]           [0, 0, 1, 0 ,0]
      [0, 2, 1, 1 ,1]           [0, 2, 1, 1 ,1]
      [1, 2, 1, 0 ,0]    =>     [0, 2, 1, 0 ,0]
      [1, 2, 0, 1 ,0]           [0, 2, 0, 0 ,0]
      [2, 2, 0, 0 ,2]           [2, 2, 0, 0 ,0]

    Use with applied_labels=[1, 2], is_onehot=False, independent=False, connectivity=2::

      [0, 0, 1, 0 ,0]           [0, 0, 1, 0 ,0]
      [0, 2, 1, 1 ,1]           [0, 2, 1, 1 ,1]
      [1, 2, 1, 0 ,0]    =>     [1, 2, 1, 0 ,0]
      [1, 2, 0, 1 ,0]           [1, 2, 0, 1 ,0]
      [2, 2, 0, 0 ,2]           [2, 2, 0, 0 ,2]

    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        applied_labels: Optional[Union[Sequence[int], int]] = None,
        is_onehot: Optional[bool] = None,
        independent: bool = True,
        connectivity: Optional[int] = None,
    ) -> None:
        """
        Args:
            applied_labels: Labels for applying the connected component analysis on.
                If given, voxels whose value is in this list will be analyzed.
                If `None`, all non-zero values will be analyzed.
            is_onehot: if `True`, treat the input data as OneHot format data, otherwise, not OneHot format data.
                default to None, which treats multi-channel data as OneHot and single channel data as not OneHot.
            independent: whether to treat ``applied_labels`` as a union of foreground labels.
                If ``True``, the connected component analysis will be performed on each foreground label independently
                and return the intersection of the largest components.
                If ``False``, the analysis will be performed on the union of foreground labels.
                default is `True`.
            connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
                Accepted values are ranging from  1 to input.ndim. If ``None``, a full
                connectivity of ``input.ndim`` is used. for more details:
                https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label.

        """
        super().__init__()
        self.applied_labels = ensure_tuple(applied_labels) if applied_labels is not None else None
        self.is_onehot = is_onehot
        self.independent = independent
        self.connectivity = connectivity

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            img: shape must be (C, spatial_dim1[, spatial_dim2, ...]).

        Returns:
            An array with shape (C, spatial_dim1[, spatial_dim2, ...]).
        """
        is_onehot = img.shape[0] > 1 if self.is_onehot is None else self.is_onehot
        if self.applied_labels is not None:
            applied_labels = self.applied_labels
        else:
            applied_labels = tuple(get_unique_labels(img, is_onehot, discard=0))

        if self.independent:
            for i in applied_labels:
                foreground = img[i] > 0 if is_onehot else img[0] == i
                mask = get_largest_connected_component_mask(foreground, self.connectivity)
                if is_onehot:
                    img[i][foreground != mask] = 0
                else:
                    img[0][foreground != mask] = 0
            return img
        if not is_onehot:  # not one-hot, union of labels
            labels, *_ = convert_to_dst_type(applied_labels, dst=img, wrap_sequence=True)
            foreground = (img[..., None] == labels).any(-1)[0]
            mask = get_largest_connected_component_mask(foreground, self.connectivity)
            img[0][foreground != mask] = 0
            return img
        # one-hot, union of labels
        foreground = (img[applied_labels, ...] == 1).any(0)
        mask = get_largest_connected_component_mask(foreground, self.connectivity)
        for i in applied_labels:
            img[i][foreground != mask] = 0
        return img


class LabelFilter:
    """
    This transform filters out labels and can be used as a processing step to view only certain labels.

    The list of applied labels defines which labels will be kept.

    Note:
        All labels which do not match the `applied_labels` are set to the background label (0).

    For example:

    Use LabelFilter with applied_labels=[1, 5, 9]::

        [1, 2, 3]         [1, 0, 0]
        [4, 5, 6]    =>   [0, 5 ,0]
        [7, 8, 9]         [0, 0, 9]
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, applied_labels: Union[Iterable[int], int]) -> None:
        """
        Initialize the LabelFilter class with the labels to filter on.

        Args:
            applied_labels: Label(s) to filter on.
        """
        self.applied_labels = ensure_tuple(applied_labels)

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Filter the image on the `applied_labels`.

        Args:
            img: Pytorch tensor or numpy array of any shape.

        Raises:
            NotImplementedError: The provided image was not a Pytorch Tensor or numpy array.

        Returns:
            Pytorch tensor or numpy array of the same shape as the input.
        """
        if not isinstance(img, (np.ndarray, torch.Tensor)):
            raise NotImplementedError(f"{self.__class__} can not handle data of type {type(img)}.")

        if isinstance(img, torch.Tensor):
            if hasattr(torch, "isin"):  # `isin` is new in torch 1.10.0
                appl_lbls = torch.as_tensor(self.applied_labels, device=img.device)
                return torch.where(torch.isin(img, appl_lbls), img, torch.tensor(0.0).to(img))
            else:
                out = self(img.detach().cpu().numpy())
                out, *_ = convert_to_dst_type(out, img)
                return out
        return np.asarray(np.where(np.isin(img, self.applied_labels), img, 0))


class FillHoles(Transform):
    r"""
    This transform fills holes in the image and can be used to remove artifacts inside segments.

    An enclosed hole is defined as a background pixel/voxel which is only enclosed by a single class.
    The definition of enclosed can be defined with the connectivity parameter::

        1-connectivity     2-connectivity     diagonal connection close-up

             [ ]           [ ]  [ ]  [ ]             [ ]
              |               \  |  /                 |  <- hop 2
        [ ]--[x]--[ ]      [ ]--[x]--[ ]        [x]--[ ]
              |               /  |  \             hop 1
             [ ]           [ ]  [ ]  [ ]

    It is possible to define for which labels the hole filling should be applied.
    The input image is assumed to be a PyTorch Tensor or numpy array with shape [C, spatial_dim1[, spatial_dim2, ...]].
    If C = 1, then the values correspond to expected labels.
    If C > 1, then a one-hot-encoding is expected where the index of C matches the label indexing.

    Note:

        The label 0 will be treated as background and the enclosed holes will be set to the neighboring class label.

        The performance of this method heavily depends on the number of labels.
        It is a bit faster if the list of `applied_labels` is provided.
        Limiting the number of `applied_labels` results in a big decrease in processing time.

    For example:

        Use FillHoles with default parameters::

            [1, 1, 1, 2, 2, 2, 3, 3]         [1, 1, 1, 2, 2, 2, 3, 3]
            [1, 0, 1, 2, 0, 0, 3, 0]    =>   [1, 1 ,1, 2, 0, 0, 3, 0]
            [1, 1, 1, 2, 2, 2, 3, 3]         [1, 1, 1, 2, 2, 2, 3, 3]

        The hole in label 1 is fully enclosed and therefore filled with label 1.
        The background label near label 2 and 3 is not fully enclosed and therefore not filled.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self, applied_labels: Optional[Union[Iterable[int], int]] = None, connectivity: Optional[int] = None
    ) -> None:
        """
        Initialize the connectivity and limit the labels for which holes are filled.

        Args:
            applied_labels: Labels for which to fill holes. Defaults to None, that is filling holes for all labels.
            connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
                Accepted values are ranging from  1 to input.ndim. Defaults to a full connectivity of ``input.ndim``.
        """
        super().__init__()
        self.applied_labels = ensure_tuple(applied_labels) if applied_labels else None
        self.connectivity = connectivity

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Fill the holes in the provided image.

        Note:
            The value 0 is assumed as background label.

        Args:
            img: Pytorch Tensor or numpy array of shape [C, spatial_dim1[, spatial_dim2, ...]].

        Raises:
            NotImplementedError: The provided image was not a Pytorch Tensor or numpy array.

        Returns:
            Pytorch Tensor or numpy array of shape [C, spatial_dim1[, spatial_dim2, ...]].
        """
        if not isinstance(img, (np.ndarray, torch.Tensor)):
            raise NotImplementedError(f"{self.__class__} can not handle data of type {type(img)}.")
        img_np, *_ = convert_data_type(img, np.ndarray)
        out_np: np.ndarray = fill_holes(img_np, self.applied_labels, self.connectivity)
        out, *_ = convert_to_dst_type(out_np, img)
        return out


class LabelToContour(Transform):
    """
    Return the contour of binary input images that only compose of 0 and 1, with Laplacian kernel
    set as default for edge detection. Typical usage is to plot the edge of label or segmentation output.

    Args:
        kernel_type: the method applied to do edge detection, default is "Laplace".

    Raises:
        NotImplementedError: When ``kernel_type`` is not "Laplace".

    """

    backend = [TransformBackends.TORCH]

    def __init__(self, kernel_type: str = "Laplace") -> None:
        if kernel_type != "Laplace":
            raise NotImplementedError('Currently only kernel_type="Laplace" is supported.')
        self.kernel_type = kernel_type

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            img: torch tensor data to extract the contour, with shape: [channels, height, width[, depth]]

        Raises:
            ValueError: When ``image`` ndim is not one of [3, 4].

        Returns:
            A torch tensor with the same shape as img, note:
                1. it's the binary classification result of whether a pixel is edge or not.
                2. in order to keep the original shape of mask image, we use padding as default.
                3. the edge detection is just approximate because it defects inherent to Laplace kernel,
                   ideally the edge should be thin enough, but now it has a thickness.

        """
        img_: torch.Tensor = convert_data_type(img, torch.Tensor)[0]
        spatial_dims = len(img_.shape) - 1
        img_ = img_.unsqueeze(0)  # adds a batch dim
        if spatial_dims == 2:
            kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
        elif spatial_dims == 3:
            kernel = -1.0 * torch.ones(3, 3, 3, dtype=torch.float32)
            kernel[1, 1, 1] = 26.0
        else:
            raise ValueError(f"{self.__class__} can only handle 2D or 3D images.")
        contour_img = apply_filter(img_, kernel)
        contour_img.clamp_(min=0.0, max=2.0)
        output, *_ = convert_to_dst_type(contour_img.squeeze(0), img)
        return output


class Ensemble:
    @staticmethod
    def get_stacked_torch(img: Union[Sequence[NdarrayOrTensor], NdarrayOrTensor]) -> torch.Tensor:
        """Get either a sequence or single instance of np.ndarray/torch.Tensor. Return single torch.Tensor."""
        if isinstance(img, Sequence) and isinstance(img[0], np.ndarray):
            img = [torch.as_tensor(i) for i in img]
        elif isinstance(img, np.ndarray):
            img = torch.as_tensor(img)
        out: torch.Tensor = torch.stack(img) if isinstance(img, Sequence) else img  # type: ignore
        return out

    @staticmethod
    def post_convert(img: torch.Tensor, orig_img: Union[Sequence[NdarrayOrTensor], NdarrayOrTensor]) -> NdarrayOrTensor:
        orig_img_ = orig_img[0] if isinstance(orig_img, Sequence) else orig_img
        out, *_ = convert_to_dst_type(img, orig_img_)
        return out


class MeanEnsemble(Ensemble, Transform):
    """
    Execute mean ensemble on the input data.
    The input data can be a list or tuple of PyTorch Tensor with shape: [C[, H, W, D]],
    Or a single PyTorch Tensor with shape: [E, C[, H, W, D]], the `E` dimension represents
    the output data from different models.
    Typically, the input data is denseUnet_3D output of segmentation task or classification task.
    And it also can support to add `weights` for the input data.

    Args:
        weights: can be a list or tuple of numbers for input data with shape: [E, C, H, W[, D]].
            or a Numpy ndarray or a PyTorch Tensor data.
            the `weights` will be added to input data from highest dimension, for example:
            1. if the `weights` only has 1 dimension, it will be added to the `E` dimension of input data.
            2. if the `weights` has 2 dimensions, it will be added to `E` and `C` dimensions.
            it's a typical practice to add weights for different classes:
            to ensemble 3 segmentation denseUnet_3D outputs, every output has 4 channels(classes),
            so the input data shape can be: [3, 4, H, W, D].
            and add different `weights` for different classes, so the `weights` shape can be: [3, 4].
            for example: `weights = [[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 1, 1]]`.

    """

    backend = [TransformBackends.TORCH]

    def __init__(self, weights: Optional[Union[Sequence[float], NdarrayOrTensor]] = None) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.float) if weights is not None else None

    def __call__(self, img: Union[Sequence[NdarrayOrTensor], NdarrayOrTensor]) -> NdarrayOrTensor:
        img_ = self.get_stacked_torch(img)
        if self.weights is not None:
            self.weights = self.weights.to(img_.device)
            shape = tuple(self.weights.shape)
            for _ in range(img_.ndimension() - self.weights.ndimension()):
                shape += (1,)
            weights = self.weights.reshape(*shape)

            img_ = img_ * weights / weights.mean(dim=0, keepdim=True)

        out_pt = torch.mean(img_, dim=0)
        return self.post_convert(out_pt, img)


class VoteEnsemble(Ensemble, Transform):
    """
    Execute vote ensemble on the input data.
    The input data can be a list or tuple of PyTorch Tensor with shape: [C[, H, W, D]],
    Or a single PyTorch Tensor with shape: [E[, C, H, W, D]], the `E` dimension represents
    the output data from different models.
    Typically, the input data is denseUnet_3D output of segmentation task or classification task.

    Note:
        This vote transform expects the input data is discrete values. It can be multiple channels
        data in One-Hot format or single channel data. It will vote to select the most common data
        between items.
        The output data has the same shape as every item of the input data.

    Args:
        num_classes: if the input is single channel data instead of One-Hot, we can't get class number
            from channel, need to explicitly specify the number of classes to vote.

    """

    backend = [TransformBackends.TORCH]

    def __init__(self, num_classes: Optional[int] = None) -> None:
        self.num_classes = num_classes

    def __call__(self, img: Union[Sequence[NdarrayOrTensor], NdarrayOrTensor]) -> NdarrayOrTensor:
        img_ = self.get_stacked_torch(img)

        if self.num_classes is not None:
            has_ch_dim = True
            if img_.ndimension() > 1 and img_.shape[1] > 1:
                warnings.warn("no need to specify num_classes for One-Hot format data.")
            else:
                if img_.ndimension() == 1:
                    # if no channel dim, need to remove channel dim after voting
                    has_ch_dim = False
                img_ = one_hot(img_, self.num_classes, dim=1)

        img_ = torch.mean(img_.float(), dim=0)

        if self.num_classes is not None:
            # if not One-Hot, use "argmax" to vote the most common class
            out_pt = torch.argmax(img_, dim=0, keepdim=has_ch_dim)
        else:
            # for One-Hot data, round the float number to 0 or 1
            out_pt = torch.round(img_)
        return self.post_convert(out_pt, img)


class ProbNMS(Transform):
    """
    Performs probability based non-maximum suppression (NMS) on the probabilities map via
    iteratively selecting the coordinate with highest probability and then move it as well
    as its surrounding values. The remove range is determined by the parameter `box_size`.
    If multiple coordinates have the same highest probability, only one of them will be
    selected.

    Args:
        spatial_dims: number of spatial dimensions of the input probabilities map.
            Defaults to 2.
        sigma: the standard deviation for gaussian filter.
            It could be a single value, or `spatial_dims` number of values. Defaults to 0.0.
        prob_threshold: the probability threshold, the function will stop searching if
            the highest probability is no larger than the threshold. The value should be
            no less than 0.0. Defaults to 0.5.
        box_size: the box size (in pixel) to be removed around the the pixel with the maximum probability.
            It can be an integer that defines the size of a square or cube,
            or a list containing different values for each dimensions. Defaults to 48.

    Return:
        a list of selected lists, where inner lists contain probability and coordinates.
        For example, for 3D input, the inner lists are in the form of [probability, x, y, z].

    Raises:
        ValueError: When ``prob_threshold`` is less than 0.0.
        ValueError: When ``box_size`` is a list or tuple, and its length is not equal to `spatial_dims`.
        ValueError: When ``box_size`` has a less than 1 value.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        spatial_dims: int = 2,
        sigma: Union[Sequence[float], float, Sequence[torch.Tensor], torch.Tensor] = 0.0,
        prob_threshold: float = 0.5,
        box_size: Union[int, Sequence[int]] = 48,
    ) -> None:
        self.sigma = sigma
        self.spatial_dims = spatial_dims
        if self.sigma != 0:
            self.filter = GaussianFilter(spatial_dims=spatial_dims, sigma=sigma)
        if prob_threshold < 0:
            raise ValueError("prob_threshold should be no less than 0.0.")
        self.prob_threshold = prob_threshold
        if isinstance(box_size, int):
            self.box_size = np.asarray([box_size] * spatial_dims)
        elif len(box_size) != spatial_dims:
            raise ValueError("the sequence length of box_size should be the same as spatial_dims.")
        else:
            self.box_size = np.asarray(box_size)
        if self.box_size.min() <= 0:
            raise ValueError("box_size should be larger than 0.")

        self.box_lower_bd = self.box_size // 2
        self.box_upper_bd = self.box_size - self.box_lower_bd

    def __call__(self, prob_map: NdarrayOrTensor):
        """
        prob_map: the input probabilities map, it must have shape (H[, W, ...]).
        """
        if self.sigma != 0:
            if not isinstance(prob_map, torch.Tensor):
                prob_map = torch.as_tensor(prob_map, dtype=torch.float)
            self.filter.to(prob_map)
            prob_map = self.filter(prob_map)

        prob_map_shape = prob_map.shape

        outputs = []
        while prob_map.max() > self.prob_threshold:
            max_idx = unravel_index(prob_map.argmax(), prob_map_shape)
            prob_max = prob_map[tuple(max_idx)]
            max_idx = max_idx.cpu().numpy() if isinstance(max_idx, torch.Tensor) else max_idx
            prob_max = prob_max.item() if isinstance(prob_max, torch.Tensor) else prob_max
            outputs.append([prob_max] + list(max_idx))

            idx_min_range = (max_idx - self.box_lower_bd).clip(0, None)
            idx_max_range = (max_idx + self.box_upper_bd).clip(None, prob_map_shape)
            # for each dimension, set values during index ranges to 0
            slices = tuple(slice(idx_min_range[i], idx_max_range[i]) for i in range(self.spatial_dims))
            prob_map[slices] = 0

        return outputs