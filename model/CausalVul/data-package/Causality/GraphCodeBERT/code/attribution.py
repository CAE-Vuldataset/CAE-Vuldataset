import torch
from captum.attr import *

def do_attribution(model, tokenizer, attribution_model, x, xp):
    model.eval()
    model.zero_grad()

    input_embeds = model.embed(input_ids=x[0], position_idx=x[1], attn_mask=x[2])

    # Test parity
    with torch.no_grad():
        pred = model(
            input_embeds,
            xpos=x[1], xatt=x[2],
            xpinput=xp[0], xppos=xp[1], xpatt=xp[2],
            return_single_prob=False,
        )

        rp_original  = model.model.get_representation(input_ids=x[0], position_idx=x[1], attn_mask=x[2])
        pred_original = model.classifier(rp_original, input_ids=xp[0], position_idx=xp[1], attn_mask=xp[2])

        assert torch.all(pred == pred_original), (pred, pred_original)
    
    kwargs = {
        "inputs": input_embeds,
        "additional_forward_args": (x[1],x[2], xp[0], xp[1], xp[2],),
        "baselines": torch.zeros_like(input_embeds, requires_grad=True),
    }


    attributions = attribution_model.attribute(**kwargs)
    
    # Summarize the attributions
    attributions = attributions.sum(dim=-1).squeeze(1)
    attributions = attributions / torch.norm(attributions)

    return {
        'attributions': attributions.detach().cpu(),
        'tokens': [tokenizer.convert_ids_to_tokens(ids) for ids in x[0]],
        'pred': pred.detach().cpu()
    }


def do_attribution_model(model, tokenizer, attribution_model, input_ids=None, position_idx=None, attn_mask=None):
    model.eval()
    model.zero_grad()

    input_embeds = model.embed(input_ids=input_ids, position_idx=position_idx, attn_mask=attn_mask)

    # Test parity
    with torch.no_grad():
        pred = model(
            inputs_embeds=input_embeds,
            position_idx=position_idx,
            attn_mask=attn_mask,
            return_single_prob=False,
        )

        pred_original  = model.model(input_ids=input_ids, position_idx=position_idx, attn_mask=attn_mask)
        assert torch.all(pred == pred_original), (pred, pred_original)
    
    kwargs = {
        "inputs": input_embeds,
        "additional_forward_args": (position_idx, attn_mask,),
        "baselines": torch.zeros_like(input_embeds, requires_grad=True),
    }

    attributions = attribution_model.attribute(**kwargs)
    
    # Summarize the attributions
    attributions = attributions.sum(dim=-1).squeeze(1)
    attributions = attributions / torch.norm(attributions)

    return {
        'attributions': attributions.detach().cpu(),
        'tokens': [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids],
        'pred': pred.detach().cpu()
    }