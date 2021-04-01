import torch

from models import PINet20

KEYPOINT_MODEL_PATH = ""
SEGMENTATION_MODEL_PATH = ""
TRANSFER_MODEL_PATH = ""

def main():
    # save keypoint_model as ONNX
    keypoint_model = PINet20.TransferModel()
    keypoint_model.load_state_dict(torch.load(KEYPOINT_MODEL_PATH))
    keypoint_model.eval()
    keypoint_dummy_input = torch.zeros(1, 3, 192, 256)
    torch.onnx.export(keypoint_model, keypoint_dummy_input, 'keypoint_model.onnx')


    # save segmentation_model as ONNX
    segmentation_model = PINet20.TransferModel()
    segmentation_model.load_state_dict(torch.load(SEGMENTATION_MODEL_PATH))
    segmentation_model.eval()
    segmentation_dummy_input = torch.zeros(1, 3, 256, 176)
    torch.onnx.export(segmentation_model, segmentation_dummy_input, 'segmentation_model.onnx')

    # save transfer_model as ONNX
    transfer_model = PINet20.TransferModel()
    transfer_model.load_state_dict(torch.load(TRANSFER_MODEL_PATH))
    transfer_model.eval()
    # 4 arguments [input_image, source_keypoint, target_keypoints, segmentation_mask]
    transfer_dummy_input = [torch.zeros(1,3, 256, 176), torch.zeros(1, 17, 256, 176), torch.zeros(1, 17, 256, 176), torch.zeros(1, 12, 256, 176)]
    torch.onnx.export(transfer_model, transfer_dummy_input, 'transfer_model.onnx')

if __name__ == '__main__':
    main()