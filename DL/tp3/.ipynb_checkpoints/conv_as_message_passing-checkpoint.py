import torch
import torch_geometric


def image_to_graph(
    image: torch.Tensor, conv2d: torch.nn.Conv2d | None = None
) -> torch_geometric.data.Data:
    """
    Converts an image tensor to a PyTorch Geometric Data object.
    COMPLETE

    Arguments:
    ----------
    image : torch.Tensor
        Image tensor of shape (C, H, W).
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None
        Is used to determine the size of the receptive field.

    Returns:
    --------
    torch_geometric.data.Data
        Graph representation of the image.
    """
    C, H, W = image.shape
    # Assumptions (remove it for the bonus)
    assert image.dim() == 3, f"Expected 3D tensor, got {image.dim()}D tensor."
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 1, "Expected padding of 1 on both sides."
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 3, "Expected kernel size of 3x3."
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."
    kernel_size = conv2d.kernel_size[0]
    
    edges_index = []
    image = image.permute(1, 2, 0)
    
    image = torch.flatten(image, 0, 1)
    pad = kernel_size // 2

    
    edges_index = []
    adj = torch.zeros(H * W, H * W, dtype=torch.float)
    edges_attr = []
    for i in range(H ):
        for j in range(W ):

            idx = i * W + j
            neighbors  = [
                (i-1, j-1), (i-1, j), (i-1, j+1),
                (i, j-1),  (i, j),       (i, j+1),
                (i+1, j-1), (i+1, j), (i+1, j+1)
            ]
            for m, (ni, nj) in enumerate(neighbors):
                if 0 <= ni < H and 0 <= nj < W:
                    edges_index.append([ni * W + nj, idx])
                    edges_attr.append(m)
    edges_attr = torch.tensor(edges_attr, dtype = torch.long)
    edges_index = torch.tensor(edges_index, dtype = torch.long).t()
    return torch_geometric.data.Data(image, edges_index, edge_attr = edges_attr)


def graph_to_image(
    data: torch.Tensor, height: int, width: int, conv2d: torch.nn.Conv2d | None = None
) -> torch.Tensor:
    """
    Converts a graph representation of an image to an image tensor.

    Arguments:
    ----------
    data : torch.Tensor
        Graph data representation of the image.
    height : int
        Height of the image.
    width : int
        Width of the image.
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None

    Returns:
    --------
    torch.Tensor
        Image tensor of shape (C, H, W).
    """
    # Assumptions (remove it for the bonus)
    assert data.dim() == 2, f"Expected 2D tensor, got {data.dim()}D tensor."
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 1, "Expected padding of 1 on both sides."
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 3, "Expected kernel size of 3x3."
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."
    C = data.shape[-1]
    image = data.reshape(height  , width , C)
    return image.permute(2, 0, 1)

class Conv2dMessagePassing(torch_geometric.nn.MessagePassing):
    """
    A Message Passing layer that simulates a given Conv2d layer.
    """

    def __init__(self, conv2d: torch.nn.Conv2d):
        # <TO IMPLEMENT>
        # Don't forget to call the parent constructor with the correct aguments
        # super().__init__(<arguments>)
        # </TO IMPLEMENT>
        super().__init__(aggr='add')
        self.in_channels = conv2d.in_channels
        self.out_channels = conv2d.out_channels
        self.conv2d = conv2d
        self.weight = conv2d.weight
        if conv2d is not None:
            assert conv2d.padding[0] == conv2d.padding[1] == 1, "Expected padding of 1 on both sides."
            assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 3, "Expected kernel size of 3x3."
            assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."
            self.kernel_size = conv2d.kernel_size[0]
            self.stride = conv2d.stride
            self.padding = conv2d.padding
    def forward(self, data):
        self.edge_index = data.edge_index

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Computes the message to be passed for each edge.
        For each edge e = (u, v) in the graph indexed by i,
        the message trough the edge e (ie from node u to node v)
        should be returned as the i-th line of the output tensor.
        (The message is phi(u, v, e) in the formalism.)
        To do this you can access the features of the source node
        in x_j[i] and the attributes of the edge in edge_attr[i].

        Arguments:
        ----------
        x_j : torch.Tensor
            The features of the souce node for each edge (of size E x in_channels).
        edge_attr : torch.Tensor
            The attributes of the edge (of size E x edge_attr_dim).

        Returns:
        --------
        torch.Tensor
            The message to be passed for each edge (of size COMPLETE)
        """
        E = x_j.shape[0]
        messages = torch.zeros((E, self.out_channels))
        for e in range(E):
            ki, kj = edge_attr[e] // 3, edge_attr[e] % 3
            source_features = x_j[e]  # Shape: (in_channels,)
            
            # get corresponding weights from the kernel
            # (out_channels, in_channels)
            kernel_weights = self.weight[:, :, ki, kj]
            
            # Shape: (out_channels,)
            message = torch.matmul(kernel_weights, source_features)
            
            messages[e] = message
        return messages