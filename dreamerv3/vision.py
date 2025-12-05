from typing import Dict, Iterable

import einops
import jax
import jax.numpy as jnp
import ninjax as nj

import embodied.jax.nets as nn

from . import rssm


f32 = jnp.float32


def make_vision_modules(config, enc_space, dec_space):
  """Factory returning encoder/decoder along with optional MAE helper."""
  enc_typ = getattr(config, 'encoder_type', 'cnn_ae')
  enc_cfg = config.enc[config.enc.typ]
  dec_cfg = config.dec[config.dec.typ]
  imgkeys = [k for k, v in enc_space.items() if len(v.shape) == 3]

  if enc_typ.startswith('vit'):
    vit_kw = dict(
        patch_size=getattr(config, 'vit_patch_size', 16),
        embed_dim=getattr(config, 'vit_embed_dim', 256),
        depth=getattr(config, 'vit_num_layers', 4),
        heads=getattr(config, 'vit_num_heads', 4),
        mlp_mult=getattr(config, 'vit_mlp_mult', 4),
        vec_layers=getattr(config, 'vit_vec_layers', 2),
        vec_units=getattr(config, 'vit_vec_units', 256),
        out_dim=getattr(config, 'vit_out_dim', getattr(config, 'vit_embed_dim', 256)),
        symlog=getattr(config, 'vit_symlog', True),
        act=getattr(config, 'vit_act', 'silu'),
        norm=getattr(config, 'vit_norm', 'rms'),
    )
    encoder = ViTEncoder(enc_space, **vit_kw, name='enc')
  else:
    encoder = rssm.Encoder(enc_space, **enc_cfg, name='enc')

  decoder = rssm.Decoder(dec_space, **dec_cfg, name='dec')

  use_mae = (
      enc_typ.endswith('mae') and
      getattr(config, 'mae_loss_scale', 0.0) > 0 and
      bool(imgkeys))
  mae = None
  if use_mae:
    mae = PatchMAELoss(
        enc_space,
        mask_ratio=getattr(config, 'mae_mask_ratio', 0.5),
        patch_size=getattr(config, 'mae_patch_size', 8),
        name='mae')

  return encoder, decoder, mae


def count_params(modules: Iterable[nj.Module]) -> int:
  total = 0
  for module in modules:
    if not module:
      continue
    values = getattr(module, 'values', {})
    total += sum(int(x.size) for x in values.values())
  return total


class ViTEncoder(nj.Module):

  patch_size: int = 16
  embed_dim: int = 256
  depth: int = 4
  heads: int = 4
  mlp_mult: int = 4
  vec_layers: int = 2
  vec_units: int = 256
  out_dim: int = 256
  act: str = 'silu'
  norm: str = 'rms'
  symlog: bool = True

  def __init__(self, obs_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    if not self.imgkeys:
      raise ValueError('ViT encoder requires at least one image observation.')
    self._patch_size = kw.get('patch_size', self.patch_size)
    self._embed_dim = kw.get('embed_dim', self.embed_dim)
    self._depth = kw.get('depth', self.depth)
    self._heads = kw.get('heads', self.heads)
    self._mlp_mult = kw.get('mlp_mult', self.mlp_mult)
    self._vec_layers = kw.get('vec_layers', self.vec_layers)
    self._vec_units = kw.get('vec_units', self.vec_units)
    self._out_dim = kw.get('out_dim', self.out_dim)
    self._act = kw.get('act', self.act)
    self._norm = kw.get('norm', self.norm)
    self._symlog = kw.get('symlog', self.symlog)

    shapes = [self.obs_space[k].shape for k in self.imgkeys]
    assert len({s[:2] for s in shapes}) == 1, shapes
    self.img_res = shapes[0][:2]
    assert all(x % self._patch_size == 0 for x in self.img_res), (
        self.img_res, self._patch_size)
    self.patch_rows = self.img_res[0] // self._patch_size
    self.patch_cols = self.img_res[1] // self._patch_size
    self.token_len = self.patch_rows * self.patch_cols + 1

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, obs, reset, training, single=False):
    bshape = reset.shape
    bdims = 1 if single else 2
    imgs = [obs[k] for k in sorted(self.imgkeys)]
    assert all(x.dtype == jnp.uint8 for x in imgs)
    x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255.0 - 0.5
    x = x.reshape((-1, *x.shape[bdims:]))
    img_feat = self._encode_image(x, training)
    img_feat = img_feat.reshape((*bshape, img_feat.shape[-1]))

    feats = [img_feat]
    if self.veckeys:
      vecspace = {k: self.obs_space[k] for k in self.veckeys}
      vecs = {k: obs[k] for k in self.veckeys}
      squish = nn.symlog if self._symlog else (lambda arr: arr)
      vec = nn.DictConcat(vecspace, 1, squish=squish)(vecs)
      vec = vec.reshape((-1, *vec.shape[bdims:]))
      for i in range(self._vec_layers):
        vec = self.sub(f'vec{i}', nn.Linear, self._vec_units)(vec)
        vec = nn.act(self._act)(self.sub(f'vecnorm{i}', nn.Norm, self._norm)(vec))
      vec = self.sub('vecproj', nn.Linear, self._embed_dim)(vec)
      vec = vec.reshape((*bshape, vec.shape[-1]))
      feats.append(vec)

    x = jnp.concatenate(feats, -1) if len(feats) > 1 else feats[0]
    flat = x.reshape((-1, x.shape[-1]))
    flat = self.sub('outproj', nn.Linear, self._out_dim)(flat)
    # Normalize output to match CNN encoder distribution
    flat = self.sub('outnorm', nn.Norm, self._norm)(flat)
    tokens = flat.reshape((*bshape, self._out_dim))
    # Replace any NaN values with zeros to prevent propagation
    tokens = jnp.where(jnp.isnan(tokens), 0.0, tokens)
    return carry, {}, tokens

  def _encode_image(self, x, training):
    B, H, W, C = x.shape
    P = self._patch_size
    assert H == self.patch_rows * P and W == self.patch_cols * P, (
        (H, W), (self.patch_rows, self.patch_cols), P)
    patches = einops.rearrange(
        x, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=P, p2=P)
    patches = nn.cast(patches)
    tokens = self.sub('patch_proj', nn.Linear, self._embed_dim)(patches)
    # Cast cls and posemb to match tokens dtype
    cls = self.value('cls', nn.init('trunc_normal'), (1, 1, self._embed_dim))
    cls = nn.cast(cls)  # Cast to compute dtype
    cls = jnp.broadcast_to(cls, (tokens.shape[0], 1, self._embed_dim))
    tokens = jnp.concatenate([cls, tokens], 1)
    posemb = self.value('posemb', nn.init('trunc_normal'), (1, self.token_len, self._embed_dim))
    posemb = nn.cast(posemb)  # Cast to compute dtype
    tokens = tokens + posemb
    ff_mult = max(1, int(self._mlp_mult))
    vit = self.sub(
        'vit', _ViTBackbone,
        self._embed_dim, self._depth, self._heads,
        ff_mult, self._act, self._norm)
    tokens = vit(tokens, training=training)
    return tokens[:, 0]


class _ViTBackbone(nj.Module):

  def __init__(self, embed_dim, depth, heads, mlp_mult, act, norm):
    self.embed_dim = embed_dim
    self.depth = depth
    self.heads = heads
    self.mlp_mult = mlp_mult
    self.act = act
    self.norm = norm

  def __call__(self, x, training):
    x = nn.cast(x)
    # Clip inputs to prevent extreme values that cause NaN
    x = jnp.clip(x, -100.0, 100.0)
    for i in range(self.depth):
      with nj.scope(f'block{i}'):
        skip = x
        x = self.sub('norm1', nn.Norm, self.norm)(x)
        x = self.sub('attn', nn.Attention, heads=self.heads, rope=False)(x, training=training)
        x = jnp.clip(x, -100.0, 100.0)  # Clip after attention
        x += skip
        skip = x
        x = self.sub('norm2', nn.Norm, self.norm)(x)
        hidden = self.sub('mlp0', nn.Linear, self.embed_dim * self.mlp_mult)(x)
        hidden = nn.act(self.act)(hidden)
        hidden = self.sub('mlp1', nn.Linear, self.embed_dim)(hidden)
        x = skip + hidden
        x = jnp.clip(x, -100.0, 100.0)  # Clip after each block
    x = self.sub('outnorm', nn.Norm, self.norm)(x)
    return x


class PatchMAELoss(nj.Module):

  mask_ratio: float = 0.5
  patch_size: int = 8

  def __init__(self, obs_space, **kw):
    self.obs_space = obs_space
    self.imgkeys = [k for k, v in obs_space.items() if len(v.shape) == 3]
    self._mask_ratio = kw.get('mask_ratio', self.mask_ratio)
    self._patch_size = kw.get('patch_size', self.patch_size)

  def __call__(self, preds: Dict[str, jnp.ndarray], targets: Dict[str, jnp.ndarray], training=True):
    if (not self.imgkeys) or self._mask_ratio <= 0 or not preds:
      return self._zeros_like(targets)
    total = 0.0
    count = 0
    for key in self.imgkeys:
      if key not in preds or key not in targets:
        continue
      total += self._loss_per_key(preds[key], targets[key])
      count += 1
    if count == 0:
      return self._zeros_like(targets)
    return total / count

  def _loss_per_key(self, pred, target):
    diff = jnp.square(f32(pred) - f32(target)).mean(-1)
    B, T, H, W = diff.shape
    patch = max(1, self._patch_size)
    assert H % patch == 0 and W % patch == 0, ((H, W), patch)
    diff = diff.reshape((B, T, H // patch, patch, W // patch, patch))
    diff = diff.mean((3, 5))
    mask_shape = diff.shape
    rand = jax.random.uniform(nj.seed(), mask_shape, f32)
    mask = (rand < self._mask_ratio).astype(diff.dtype)
    denom = jnp.maximum(mask.sum((-2, -1)), 1.0)
    loss = (diff * mask).sum((-2, -1)) / denom
    return loss

  def _zeros_like(self, targets: Dict[str, jnp.ndarray]):
    if targets:
      ref = next(iter(targets.values()))
      return jnp.zeros(ref.shape[:2], f32)
    return jnp.zeros((1, 1), f32)

