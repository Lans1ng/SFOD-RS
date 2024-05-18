import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as trans
import os
CLASSES = ['airplane', 'airport', 'baseball field','basketball court', 
           'bridge', 'chimney', 'dam', 'Expressway-Service-area',
           'Expressway-toll-station', 'golf field', 'ground track field', 
           'harbor', 'overpass', 'ship','stadium', 'storage tank', 'tennis court',
           'train station', 'vehicle', 'windmill']
save_img = False

def obb2xyxy(rbboxes):
    w = rbboxes[:, 2::5]
    h = rbboxes[:, 3::5]
    a = rbboxes[:, 4::5]
    cosa = np.abs(np.cos(a))
    sina = np.abs(np.sin(a))
    hbbox_w = cosa * w + sina * h
    hbbox_h = sina * w + cosa * h
    dx = rbboxes[..., 0]
    dy = rbboxes[..., 1]
    dw = hbbox_w.reshape(-1)
    dh = hbbox_h.reshape(-1)
    x1 = dx - dw / 2
    y1 = dy - dh / 2
    x2 = dx + dw / 2
    y2 = dy + dh / 2
    xyxy_array = np.stack((x1, y1, x2, y2), -1)

    return xyxy_array


class CGA:
    def __init__(self, class_names, model='RN50x64', templates = 'an aerial image of a {}'):
        super().__init__()
        self.save_path = '_clip_img'
        self.device = torch.cuda.current_device()

        self.expand_ratio = 0.4
        
        # CLIP configs
        import clip
        self.class_names = class_names

        self.clip, self.preprocess = clip.load(model, device=self.device)
        self.prompts = clip.tokenize([
            templates.format(cls_name) 
            for cls_name in self.class_names
        ]).to(self.device)
        with torch.no_grad():
            self.text_features = self.clip.encode_text(self.prompts)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def load_image_by_box(self, img_path, boxes, scores, labels):
        image = Image.open(img_path).convert("RGB")
        image_list = []
        probs_list = []
        ori_image_list = []
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = box
            h, w = y2 - y1, x2 - x1
            x1 = max(0, x1 - w * self.expand_ratio)
            y1 = max(0, y1 - h * self.expand_ratio)
            x2 = x2 + w * self.expand_ratio
            y2 = y2 + h * self.expand_ratio
            sub_image = image.crop((int(x1), int(y1), int(x2), int(y2)))
            if save_img:
                label_ = CLASSES[label]
                sub_image.save(os.path.join(self.save_path, f"sub_image_{i}_{score}_{label_}.jpg"))

            ori_image_list.append(sub_image)
            sub_image = self.preprocess(sub_image).to(self.device)
            image_list.append(sub_image)
        return torch.stack(image_list), ori_image_list
        
    @torch.no_grad()
    def __call__(self, img_path, boxes, scores, labels):
        images,ori_image_list = self.load_image_by_box(img_path, boxes, scores, labels)
        with torch.no_grad():
            image_features = self.clip.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        logits_per_image = (100*image_features @ self.text_features.T).softmax(dim=-1).cpu().numpy()
        return logits_per_image, ori_image_list

class TestMixins:
    def __init__(self):
        self.cga = None

    def refine_test(self, results, img_metas):
        if not hasattr(self, 'cga'):

            self.cga= CGA(CLASSES, model='RN50x64')
            self.exclude_ids = [7,8,11]

        boxes_list, scores_list, labels_list = [], [], []
        for cls_id, result in enumerate(results[0]):
            if len(result) == 0:
                continue
        
            result_ = obb2xyxy(result)                

            boxes_list.append(result_[:, :4])
            scores_list.append(result[:, -1])

            labels_list.append([cls_id] * len(result))
        if len(boxes_list) == 0:
            return results
        
        boxes_list = np.concatenate(boxes_list, axis=0)

        scores_list = np.concatenate(scores_list, axis=0)
        labels_list = np.concatenate(labels_list, axis=0)

        logits, images = self.cga(img_metas[0]['filename'], boxes_list, scores_list, labels_list)

        for i, prob in enumerate(logits):
            
            if labels_list[i] != np.argmax(prob):
                if labels_list[i] not in self.exclude_ids:
                    scores_list[i] = scores_list[i] * 0.7 + prob[labels_list[i]] * 0.3
            else:
                pass
        j = 0
        for i in range(len(results[0])):
            num_dets = len(results[0][i])
            if num_dets == 0:
                continue
            for k in range(num_dets):
                results[0][i][k, -1] = scores_list[j]
                j += 1
        
        return results

