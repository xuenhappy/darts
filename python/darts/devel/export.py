import torch
import sys
if __name__ == "__main__":
    model = torch.load(sys.argv[1], map_location=torch.device('cpu')).ner.cpu()
    model.train(False)
    model.export2onnx()
