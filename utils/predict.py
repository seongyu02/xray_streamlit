import torch
import torch.nn.functional as F


STAGE2_CLASSES = ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]


def predict_stage1(model, input_tensor, threshold=0.5):
    with torch.no_grad():
        output = model(input_tensor)

        if output.shape[1] == 1:
            abnormal_prob = torch.sigmoid(output)[0][0].item()
        else:
            probs = F.softmax(output, dim=1)[0]
            abnormal_prob = probs[1].item()

    if abnormal_prob >= threshold:
        return "ABNORMAL", abnormal_prob
    else:
        return "NORMAL", 1.0 - abnormal_prob


def predict_stage2(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]

    class_idx = torch.argmax(probs).item()
    class_name = STAGE2_CLASSES[class_idx]
    confidence = probs[class_idx].item()

    prob_dict = {
        STAGE2_CLASSES[i]: probs[i].item()
        for i in range(len(STAGE2_CLASSES))
    }

    return class_name, confidence, prob_dict