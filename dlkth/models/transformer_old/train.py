import torch
import copy


@torch.no_grad()
def estimate_loss(model, get_batch, train_data, val_data, eval_iters):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_model(
    model,
    optimizer,
    train_data,
    val_data,
    get_batch,
    estimate_loss,
    max_iters,
    eval_interval,
    patience=3,
):
    best_val_loss = float("inf")
    best_model = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(
                model, get_batch, train_data, val_data, eval_interval
            )
            val_loss = losses["val"]
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

            # early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at step {iter}")
                    break

        xb, yb = get_batch("train", train_data, val_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.load_state_dict(best_model)
