import timm
model = timm.create_model("convnext_tiny", pretrained=True)
print(model.stages[3].blocks[-1])