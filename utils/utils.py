from torchinfo import summary
def print_model_info(model, input_size, name):
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {name}")
    # print(f"\n🧠 Total parameters: {total_params:,}")
    # print(f"🎯 Trainable parameters: {trainable_params:,}")

    # Show model summary
    print("\n📋 Model Summary:\n")
    summary(model,
            input_size=input_size,
            depth=2,
            verbose=1,
            col_names=["input_size", "output_size", "num_params", "kernel_size"])