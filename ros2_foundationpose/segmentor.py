import torch
import numpy as np
# from segment_anything.utils.amg import remove_small_regions 会卡
from ros2_foundationpose.model_config import SAMConfig

class Segmentor:
    def __init__(self,
                 model_type="vit_t",
                 mode="prompt",
                 device="cuda:0",
                 area_threshold=400,
                 refine_mode="islands", # "holes" or "islands"
                 refine_mask=True):
        
        self.sam_config = SAMConfig.from_args(model_type, mode, device)
        self.sam_predictor = self.sam_config.get_predictor()
        
        self.area_threshold = area_threshold
        self.refine_mode = refine_mode
        self.refine_mask = refine_mask
            
    @torch.no_grad()
    def interactive_predict(self, prompts, mode, multimask=True):    
        
        if mode == 'point':
            masks, scores, logits = self.sam_predictor.predict(point_coords=prompts['point_coords'], 
                                point_labels=prompts['point_modes'], 
                                multimask_output=multimask)
        elif mode == 'mask':
            masks, scores, logits = self.sam_predictor.predict(mask_input=prompts['mask_prompt'], 
                                multimask_output=multimask)
        elif mode == 'point_mask':
            masks, scores, logits = self.sam_predictor.predict(point_coords=prompts['point_coords'], 
                                point_labels=prompts['point_modes'], 
                                mask_input=prompts['mask_prompt'], 
                                multimask_output=multimask)
                                
        return masks, scores, logits
        
    @torch.no_grad()
    def segment_with_click(self, origin_frame, coords, modes, multimask=True):
        '''
            
            return: 
                mask: one-hot 
        '''
        assert self.sam_config.mode == "prompt"
        self.sam_predictor.set_image(origin_frame)

        prompts = {
            'point_coords': coords,
            'point_modes': modes,
        }
        masks, scores, logits = self.interactive_predict(prompts, 'point', multimask)
        mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
        prompts = {
            'point_coords': coords,
            'point_modes': modes,
            'mask_prompt': logit[None, :, :]
        }
        masks, scores, logits = self.interactive_predict(prompts, 'point_mask', multimask)
        mask = masks[np.argmax(scores)]

        return mask.astype(np.uint8)

    def segment_with_box(self, origin_frame, xyxy):
        assert self.sam_config.mode == "prompt"
        self.sam_predictor.set_image(origin_frame)
        # coord = np.array([[int((bbox[1][0] - bbox[0][0]) / 2.),  int((bbox[1][1] - bbox[0][1]) / 2)]])
        # point_label = np.array([1])

        masks, scores, logits = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=xyxy,
            multimask_output=True
        )
        mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]

        if self.refine_mask:
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=xyxy,
                mask_input=logit[None, :, :],
                multimask_output=True
            )
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
            # mask, _ = remove_small_regions(mask, self.area_threshold, mode=self.refine_mode)
            return mask, logit
    
    def segment_with_boxes(self, origin_frame, boxed):
        masks = []
        logits = []
        for xyxy in boxed:
            mask, logit = self.segment_with_box(origin_frame, xyxy)
            masks.append(mask)
            logits.append(logit)
            
        return masks, logits
    
    def segment_auto(self, origin_frame):
        masks = self.sam_predictor.generate(origin_frame)  # dict of masks
        masks = [mask["segmentation"] for mask in masks]  # list of [H, W]
        return masks
