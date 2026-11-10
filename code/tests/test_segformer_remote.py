"""Runnable B2 checkpoint, geometry and gradient check without MMCV."""
from pathlib import Path

import torch
from torch.nn import functional as F

from core.models.segformer import CHANNELS, DEPTHS, HEADS, RATIOS, build_segformer_b2, load_imagenet_b2


def openmmlab_reference_features(state, image):
    """Independent FP32 forward using the operations in the pinned MMCV MiT."""
    outputs = []
    for stage, (channels, depth, heads, ratio) in enumerate(zip(CHANNELS, DEPTHS, HEADS, RATIOS)):
        prefix = f'layers.{stage}.'
        image = F.conv2d(image, state[prefix+'0.projection.weight'], state[prefix+'0.projection.bias'],
                         stride=4 if stage == 0 else 2, padding=3 if stage == 0 else 1)
        height, width = image.shape[-2:]
        sequence = F.layer_norm(image.flatten(2).transpose(1, 2), (channels,),
                                state[prefix+'0.norm.weight'], state[prefix+'0.norm.bias'], 1e-6)
        for block in range(depth):
            key = prefix+f'1.{block}.'
            q = F.layer_norm(sequence, (channels,), state[key+'norm1.weight'], state[key+'norm1.bias'], 1e-6)
            kv = q
            if ratio > 1:
                reduced = F.conv2d(q.transpose(1, 2).reshape(-1, channels, height, width),
                                   state[key+'attn.sr.weight'], state[key+'attn.sr.bias'], stride=ratio)
                kv = F.layer_norm(reduced.flatten(2).transpose(1, 2), (channels,),
                                  state[key+'attn.norm.weight'], state[key+'attn.norm.bias'], 1e-6)
            attention, _ = F.multi_head_attention_forward(
                q.transpose(0, 1), kv.transpose(0, 1), kv.transpose(0, 1),
                embed_dim_to_check=channels, num_heads=heads,
                in_proj_weight=state[key+'attn.attn.in_proj_weight'],
                in_proj_bias=state[key+'attn.attn.in_proj_bias'], bias_k=None, bias_v=None,
                add_zero_attn=False, dropout_p=0.,
                out_proj_weight=state[key+'attn.attn.out_proj.weight'],
                out_proj_bias=state[key+'attn.attn.out_proj.bias'], training=False,
                key_padding_mask=None, need_weights=False, attn_mask=None)
            sequence = sequence + attention.transpose(0, 1)
            hidden = F.layer_norm(sequence, (channels,), state[key+'norm2.weight'], state[key+'norm2.bias'], 1e-6)
            hidden = hidden.transpose(1, 2).reshape(-1, channels, height, width)
            hidden = F.conv2d(hidden, state[key+'ffn.layers.0.weight'], state[key+'ffn.layers.0.bias'])
            hidden = F.conv2d(hidden, state[key+'ffn.layers.1.weight'], state[key+'ffn.layers.1.bias'],
                              padding=1, groups=hidden.shape[1])
            hidden = F.gelu(hidden)
            hidden = F.conv2d(hidden, state[key+'ffn.layers.4.weight'], state[key+'ffn.layers.4.bias'])
            sequence = sequence + hidden.flatten(2).transpose(1, 2)
        sequence = F.layer_norm(sequence, (channels,), state[prefix+'2.weight'], state[prefix+'2.bias'], 1e-6)
        image = sequence.transpose(1, 2).reshape(-1, channels, height, width)
        outputs.append(image)
    return outputs


def test_b2():
    encoder, decoder = build_segformer_b2(6)
    checkpoint = Path('pretrain/mit_b2_20220624-66e8bf70.pth')
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        load_imagenet_b2(encoder, state)
        encoder.eval()
        with torch.no_grad():
            fixed = torch.linspace(-1, 1, 3*65*67).reshape(1, 3, 65, 67)
            actual = encoder(fixed)[1]
            expected = openmmlab_reference_features(state.get('state_dict', state), fixed)
            errors = [float((a-b).abs().max()) for a, b in zip(actual, expected)]
            assert max(errors) < 1e-4, errors
            print('OpenMMLab MiT-B2 feature max abs errors:', errors)
            decoder.eval()
            projected = []
            for index in (4, 3, 2, 1):
                feature = actual[index-1]
                linear = getattr(decoder, f'linear_c{index}').proj
                value = F.linear(feature.flatten(2).transpose(1, 2), linear.weight, linear.bias)
                value = value.transpose(1, 2).reshape(1, 768, *feature.shape[-2:])
                projected.append(F.interpolate(value, size=actual[0].shape[-2:], mode='bilinear',
                                               align_corners=False) if index != 1 else value)
            fused = F.conv2d(torch.cat(projected, dim=1), decoder.linear_fuse[0].weight)
            bn = decoder.linear_fuse[1]
            fused = F.relu(F.batch_norm(fused, bn.running_mean, bn.running_var, bn.weight,
                                        bn.bias, training=False, eps=bn.eps))
            reference_logits = F.conv2d(fused, decoder.linear_pred.weight, decoder.linear_pred.bias)
            reference_logits = F.interpolate(reference_logits, size=fixed.shape[-2:],
                                             mode='bilinear', align_corners=False)
            logit_error = float((decoder(actual, fixed.shape[-2:])-reference_logits).abs().max())
            assert logit_error < 1e-5, logit_error
            print('SegFormer MLP-768 head max abs error:', logit_error)
    encoder.eval(); decoder.train()
    image = torch.randn(2, 3, 65, 67)
    features = encoder(image)[1]
    assert tuple(t.shape[1] for t in features) == CHANNELS
    assert [t.shape[-2:] for t in features] == [(17, 17), (9, 9), (5, 5), (3, 3)]
    logits = decoder(features, image.shape[-2:])
    assert logits.shape == (2, 6, 65, 67)
    logits.square().mean().backward()
    assert encoder.backbone.patch_embed1.proj.weight.grad.abs().sum() > 0
    assert decoder.linear_c4.proj.weight.grad.abs().sum() > 0


if __name__ == '__main__':
    test_b2()
    print('MiT-B2 checkpoint, odd geometry and backward OK')
