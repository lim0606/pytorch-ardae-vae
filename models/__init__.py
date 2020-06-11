# dae
from models.dae.mlp import DAE as MLPDAE
from models.dae.mlp import ConditionalDAE as MLPCDAE

from models.resdae.mlp import DAE as MLPResDAE
from models.resdae.mlp import ARDAE as MLPResARDAE
from models.resdae.mlp import ConditionalDAE as MLPResCDAE
from models.resdae.mlp import ConditionalARDAE as MLPResCARDAE

from models.graddae.mlp import DAE as MLPGradDAE
from models.graddae.mlp import ARDAE as MLPGradARDAE
from models.graddae.mlp import ConditionalDAE as MLPGradCDAE
from models.graddae.mlp import ConditionalARDAE as MLPGradCARDAE

# vae
from models.vae.toy import VAE as ToyVAE
from models.vae.mnist import VAE as MNISTVAE
from models.vae.conv import VAE as MNISTConvVAE
from models.vae.resconv import VAE as MNISTResConvVAE
from models.vae.auxtoy import VAE as ToyAuxVAE
from models.vae.auxmnist import VAE as MNISTAuxVAE
from models.vae.auxconv import VAE as MNISTConvAuxVAE
from models.vae.auxresconv import VAE as MNISTResConvAuxVAE

# ivae
from models.ivae.toy import ImplicitPosteriorVAE as ToyIPVAE
from models.ivae.mnist import ImplicitPosteriorVAE as MNISTIPVAE
from models.ivae.conv import ImplicitPosteriorVAE as ConvIPVAE
from models.ivae.resconv import ImplicitPosteriorVAE as ResConvIPVAE
from models.ivae.auxtoy import ImplicitPosteriorVAE as ToyAuxIPVAE
from models.ivae.auxmnist import ImplicitPosteriorVAE as MNISTAuxIPVAE
from models.ivae.auxconv import ImplicitPosteriorVAE as MNISTConvAuxIPVAE
from models.ivae.auxresconv import ImplicitPosteriorVAE as MNISTResConvAuxIPVAE
from models.ivae.auxresconv2 import ImplicitPosteriorVAE as MNISTResConvAuxIPVAEClipped
