import torch 
import torch.nn as nn 


from ddp_lab.models.resnet import build_resnet18_cifar 

SEED = 42 

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 78)
print("ResNet18 CIFAR-100 Model Test")
print("=" * 78)

print(f"Device               : {device}")

if device.type == "cuda":
    print(
        f"GPU   :"
        f"{torch.cuda.get_device_name(0)}"
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

model = build_resnet18_cifar(num_classes=100)
model = model.to(device)

# parameters count 
total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)
trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print()
print("MODEL")
print("-" * 78)

print(
    f"Total parameters     : "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters : "
    f"{trainable_parameters:,}"
)

## Synth CIFAR-style batch
## shape:[batch, channels, height, width], CIFAR: channel=3, height=32, width=32

batch_size = 8 
images = torch.randn(batch_size, 3, 32, 32, device=device,)

# random class labels 
labels = torch.randint(low=0, high=100, size=(batch_size,), device=device)

print()
print("INPUT")
print("-" * 78)

print(
    f"Images shape         : "
    f"{tuple(images.shape)}"
)

print(
    f"Labels shape         : "
    f"{tuple(labels.shape)}"
)

# forward pass 

logits = model(images)

print()
print("FORWARD PASS")
print("-" * 78)

print(
    f"Logits shape         : "
    f"{tuple(logits.shape)}"
)

expected_shape = (batch_size, 100,)

assert (
    tuple(logits.shape)
    == expected_shape
), (
    f"Expected {expected_shape}, "
    f"got {tuple(logits.shape)}"
)


print(
    "Output shape         : PASSED"
)


assert torch.isfinite(
    logits
).all()


print(
    "Finite logits        : PASSED"
)

# cross entropy loss 
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels,)

print()
print("LOSS")
print("-" * 78)

print(
    f"Cross-entropy loss   : "
    f"{loss.item():.6f}"
)


assert torch.isfinite(
    loss
)


print(
    "Finite loss          : PASSED"
)

# backward pass 
loss.backward()

parameters_with_grad = 0 
finite_gradient_tensors = 0 

for parameter in model.parameters():
    if parameter.grad is not None:
        parameters_with_grad += 1

        if torch.isfinite(parameter.grad).all():
            finite_gradient_tensors += 1


print()
print("BACKWARD PASS")
print("-" * 78)

print(
    f"Tensors with grad    : "
    f"{parameters_with_grad}"
)

print(
    f"Finite grad tensors  : "
    f"{finite_gradient_tensors}"
)


assert (
    parameters_with_grad
    > 0
)


assert (
    finite_gradient_tensors
    == parameters_with_grad
)


print(
    "Gradient check       : PASSED"
)


# gpu memory 

if device.type == "cuda":

    torch.cuda.synchronize()

    allocated_gb = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved_gb = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    peak_gb = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )


    print()
    print("GPU MEMORY")
    print("-" * 78)

    print(
        f"Allocated            : "
        f"{allocated_gb:.3f} GiB"
    )

    print(
        f"Reserved             : "
        f"{reserved_gb:.3f} GiB"
    )

    print(
        f"Peak allocated       : "
        f"{peak_gb:.3f} GiB"
    )


print()
print("=" * 78)
print("RESNET18 MODEL TEST PASSED")
print("=" * 78)