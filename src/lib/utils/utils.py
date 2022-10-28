import time
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def inference_time(model: torch.nn.Module,
                   device,
                   input_size: torch.Size,
                   num_runs: int,
                   num_warmup: int):
    """
    Implements benchmarking of the inference time of a given model on a given device for an input size.
    Args:
        model: model that is evaluated
        device: device for execution
        input_size: shape of input tensor
        num_runs: number of runs over which the inference time is averaged
        num_warmup: number of warmup runs beforehand

    Returns: average inference time in nano seconds
    """
    model.eval()
    model.to(device)
    inp = torch.randn(size=input_size).to(device).float()
    with torch.no_grad():
        for _ in range(num_warmup):
            model(inp)
            torch.cuda.synchronize(device)

    with torch.no_grad():
        t0 = time.time_ns()
        for _ in range(num_runs):
            model(inp)
            torch.cuda.synchronize(device)
        t1 = time.time_ns()

    return (t1 - t0) / num_runs

