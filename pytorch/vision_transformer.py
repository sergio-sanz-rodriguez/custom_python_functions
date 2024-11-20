import torch
import torchvision
from torch import nn

class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """
    # Initialize the class with appropriate variables
    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()

        # Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2, # only flatten the feature map dimensions into a single vector. dim=0 is the batch size, dim=1 is the embedding dimension.
                                  end_dim=3)
        
        # Class token embedding
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim),
                                        requires_grad=True)
        
        # Class position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, (patch_size**2)+1, embedding_dim),
                                          requires_grad=True)
        
        # Variables
        self.patch_size = patch_size

    # Define the forward method
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {patch_size}"

        # Perform the forward pass
        x_patched = self.patcher(x)

        # Flatten the linear transformed patches
        x_flattened = self.flatten(x_patched).permute(0, 2, 1) # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]

        # Create class token and prepend
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)
        x_flattened_token = torch.cat((class_token, x_flattened), dim=1)

        # Create position embedding
        pos_embedding = self.pos_embedding.expand(batch_size, -1, -1)
        x_flattened_token_pos = x_flattened_token + pos_embedding

        return x_flattened_token_pos




    
