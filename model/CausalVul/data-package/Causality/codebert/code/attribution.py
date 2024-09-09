import torch
from captum.attr import *

def do_attribution(model, tokenizer, attribution_model, input_ids, xp_ids):
    model.eval()
    model.zero_grad()

    input_embeds = model.embed(input_ids)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    # Test parity
    with torch.no_grad():
        pred = model(
            input_embeds,
            attention_mask=attention_mask,
            xp_ids=xp_ids,
            return_single_prob=False,
        )

        rp_original  = model.model.get_representation(input_ids)
        pred_original = model.classifier(rp_original, xp_ids)

        assert torch.all(pred == pred_original), (pred, pred_original)
    
    kwargs = {
        "inputs": input_embeds,
        "additional_forward_args": (attention_mask, xp_ids),
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


def do_attribution_model(model, tokenizer, attribution_model, input_ids):
    model.eval()
    model.zero_grad()

    input_embeds = model.embed(input_ids)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    # Test parity
    with torch.no_grad():
        pred = model(
            input_embeds,
            attention_mask=attention_mask,
            return_single_prob=False,
        )

        pred_original  = model.model(input_ids)
        assert torch.all(pred == pred_original), (pred, pred_original)
    
    kwargs = {
        "inputs": input_embeds,
        "additional_forward_args": (attention_mask,),
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