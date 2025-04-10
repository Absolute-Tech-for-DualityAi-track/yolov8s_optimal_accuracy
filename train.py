EPOCHS = 300
MOSAIC = 0.8
OPTIMIZER = 'AdamW'
MOMENTUM = 0.95
LR0 = 0.001
LRF = 0.0001
SINGLE_CLS = False
BATCH_SIZE = 16
IMG_SIZE = 640  # Changed from 1024 to 640 as per args

import argparse
from ultralytics import YOLO
import os
import sys
import torch
import time

# Enable CUDA optimizations
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = True
if hasattr(torch.backends, 'cuda'):
    torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    # epochs
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    # mosaic
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation')
    # optimizer
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer')
    # momentum
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum')
    # lr0
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    # lrf
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    # single_cls
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Single class training')
    # batch size
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='Batch size')
    # image size
    parser.add_argument('--imgsz', type=int, default=IMG_SIZE, help='Image size')
    # model
    parser.add_argument('--model', type=str, default='D:/HackByte_Dataset_vol_1/yolov8s.pt', help='Model to use')
    # data
    parser.add_argument('--data', type=str, default='D:/HackByte_Dataset_vol_1/yolo_params.yaml', help='Data configuration file')
    # device
    parser.add_argument('--device', type=str, default='0', help='Device to use (empty for auto, cpu, 0, 1, etc.)')
    # patience
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    # save_period
    parser.add_argument('--save_period', type=int, default=10, help='Save checkpoint every X epochs')
    # cache
    parser.add_argument('--cache', type=str, default='ram', help='Cache images in RAM')
    # workers
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    # project
    parser.add_argument('--project', type=str, default='runs/train', help='Project name')
    # name
    parser.add_argument('--name', type=str, default='high_accuracy_model', help='Experiment name')
    # exist_ok
    parser.add_argument('--exist_ok', type=bool, default=True, help='Overwrite existing experiment')
    # pretrained
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained weights')
    # verbose
    parser.add_argument('--verbose', type=bool, default=True, help='Verbose output')
    # seed
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    # deterministic
    parser.add_argument('--deterministic', type=bool, default=True, help='Deterministic training')
    # rect
    parser.add_argument('--rect', type=bool, default=False, help='Rectangular training')
    # cos_lr
    parser.add_argument('--cos_lr', type=bool, default=True, help='Use cosine learning rate')
    # close_mosaic
    parser.add_argument('--close_mosaic', type=int, default=100, help='Disable mosaic after X epochs')
    # resume
    parser.add_argument('--resume', type=bool, default=False, help='Resume training from last checkpoint')
    # amp
    parser.add_argument('--amp', type=bool, default=True, help='Automatic mixed precision')
    # fraction
    parser.add_argument('--fraction', type=float, default=1.0, help='Fraction of dataset to use')
    # profile
    parser.add_argument('--profile', type=bool, default=False, help='Profile training')
    # freeze
    parser.add_argument('--freeze', type=int, default=None, help='Freeze layers')
    # multi_scale
    parser.add_argument('--multi_scale', type=bool, default=False, help='Multi-scale training')
    # overlap_mask
    parser.add_argument('--overlap_mask', type=bool, default=True, help='Masks should overlap during training')
    # mask_ratio
    parser.add_argument('--mask_ratio', type=int, default=4, help='Mask downsample ratio')
    # dropout
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout regularization')
    # val
    parser.add_argument('--val', type=bool, default=True, help='Validate during training')
    # split
    parser.add_argument('--split', type=str, default='val', help='Validation split')
    # save_json
    parser.add_argument('--save_json', type=bool, default=False, help='Save results to JSON')
    # save_hybrid
    parser.add_argument('--save_hybrid', type=bool, default=False, help='Save hybrid version of labels')
    # conf
    parser.add_argument('--conf', type=float, default=None, help='Confidence threshold')
    # iou
    parser.add_argument('--iou', type=float, default=0.7, help='NMS IoU threshold')
    # max_det
    parser.add_argument('--max_det', type=int, default=300, help='Maximum detections per image')
    # half
    parser.add_argument('--half', type=bool, default=False, help='Use half precision')
    # dnn
    parser.add_argument('--dnn', type=bool, default=False, help='Use OpenCV DNN for ONNX inference')
    # plots
    parser.add_argument('--plots', type=bool, default=True, help='Generate plots for analysis')
    # source
    parser.add_argument('--source', type=str, default=None, help='Source for inference')
    # vid_stride
    parser.add_argument('--vid_stride', type=int, default=1, help='Video frame stride')
    # stream_buffer
    parser.add_argument('--stream_buffer', type=bool, default=False, help='Buffer all stream frames')
    # visualize
    parser.add_argument('--visualize', type=bool, default=False, help='Visualize features')
    # augment
    parser.add_argument('--augment', type=bool, default=False, help='Apply augmented inference')
    # agnostic_nms
    parser.add_argument('--agnostic_nms', type=bool, default=False, help='Class-agnostic NMS')
    # classes
    parser.add_argument('--classes', type=list, default=None, help='Filter by class')
    # retina_masks
    parser.add_argument('--retina_masks', type=bool, default=False, help='Use retina masks')
    # embed
    parser.add_argument('--embed', type=list, default=None, help='Embed features')
    # show
    parser.add_argument('--show', type=bool, default=False, help='Show results')
    # save_frames
    parser.add_argument('--save_frames', type=bool, default=False, help='Save video frames')
    # save_txt
    parser.add_argument('--save_txt', type=bool, default=False, help='Save results to txt')
    # save_conf
    parser.add_argument('--save_conf', type=bool, default=False, help='Save confidences in --save-txt labels')
    # save_crop
    parser.add_argument('--save_crop', type=bool, default=False, help='Save cropped prediction boxes')
    # show_labels
    parser.add_argument('--show_labels', type=bool, default=True, help='Show labels')
    # show_conf
    parser.add_argument('--show_conf', type=bool, default=True, help='Show confidences')
    # show_boxes
    parser.add_argument('--show_boxes', type=bool, default=True, help='Show boxes')
    # line_width
    parser.add_argument('--line_width', type=int, default=None, help='Line width')
    # format
    parser.add_argument('--format', type=str, default='torchscript', help='Export format')
    # keras
    parser.add_argument('--keras', type=bool, default=False, help='Export to Keras')
    # optimize
    parser.add_argument('--optimize', type=bool, default=False, help='Optimize model for inference')
    # int8
    parser.add_argument('--int8', type=bool, default=False, help='INT8 quantization')
    # dynamic
    parser.add_argument('--dynamic', type=bool, default=False, help='Dynamic ONNX axes')
    # simplify
    parser.add_argument('--simplify', type=bool, default=True, help='Simplify ONNX model')
    # opset
    parser.add_argument('--opset', type=int, default=None, help='ONNX opset version')
    # workspace
    parser.add_argument('--workspace', type=int, default=None, help='ONNX workspace size')
    # nms
    parser.add_argument('--nms', type=bool, default=False, help='NMS in ONNX')
    # warmup_epochs
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs')
    # warmup_momentum
    parser.add_argument('--warmup_momentum', type=float, default=0.8, help='Warmup momentum')
    # warmup_bias_lr
    parser.add_argument('--warmup_bias_lr', type=float, default=0.1, help='Warmup bias learning rate')
    # box
    parser.add_argument('--box', type=float, default=7.5, help='Box loss gain')
    # cls
    parser.add_argument('--cls', type=float, default=0.6, help='Cls loss gain')
    # dfl
    parser.add_argument('--dfl', type=float, default=1.5, help='DFL loss gain')
    # pose
    parser.add_argument('--pose', type=float, default=12.0, help='Pose loss gain')
    # kobj
    parser.add_argument('--kobj', type=float, default=1.0, help='Keypoint obj loss gain')
    # nbs
    parser.add_argument('--nbs', type=int, default=64, help='Nominal batch size')
    # hsv_h
    parser.add_argument('--hsv_h', type=float, default=0.015, help='HSV-Hue augmentation')
    # hsv_s
    parser.add_argument('--hsv_s', type=float, default=0.7, help='HSV-Saturation augmentation')
    # hsv_v
    parser.add_argument('--hsv_v', type=float, default=0.4, help='HSV-Value augmentation')
    # degrees
    parser.add_argument('--degrees', type=float, default=10.0, help='Random rotation')
    # translate
    parser.add_argument('--translate', type=float, default=0.2, help='Random translation')
    # scale
    parser.add_argument('--scale', type=float, default=0.5, help='Random scaling')
    # shear
    parser.add_argument('--shear', type=float, default=0.5, help='Random shear')
    # perspective
    parser.add_argument('--perspective', type=float, default=0.0005, help='Random perspective')
    # flipud
    parser.add_argument('--flipud', type=float, default=0.1, help='Random flip up-down')
    # fliplr
    parser.add_argument('--fliplr', type=float, default=0.5, help='Random flip left-right')
    # bgr
    parser.add_argument('--bgr', type=float, default=0.0, help='BGR augmentation')
    # mixup
    parser.add_argument('--mixup', type=float, default=0.3, help='Image mixup')
    # copy_paste
    parser.add_argument('--copy_paste', type=float, default=0.3, help='Copy-paste augmentation')
    # copy_paste_mode
    parser.add_argument('--copy_paste_mode', type=str, default='flip', help='Copy-paste mode')
    # auto_augment
    parser.add_argument('--auto_augment', type=str, default='randaugment', help='Auto augmentation')
    # erasing
    parser.add_argument('--erasing', type=float, default=0.4, help='Random erasing')
    # crop_fraction
    parser.add_argument('--crop_fraction', type=float, default=1.0, help='Crop fraction')
    # cfg
    parser.add_argument('--cfg', type=str, default=None, help='Model configuration')
    # tracker
    parser.add_argument('--tracker', type=str, default='botsort.yaml', help='Tracker configuration')
    # save_dir
    parser.add_argument('--save_dir', type=str, default='runs\\train\\high_accuracy_model', help='Save directory')
    
    args = parser.parse_args()
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    
    print("Starting high-accuracy YOLOv8 training...")
    start_time = time.time()
    
    # Determine device
    if args.device:
        device = args.device
    else:
        device = 0 if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {'CUDA' if torch.cuda.is_available() and device != 'cpu' else 'CPU'}")
    if torch.cuda.is_available() and device != 'cpu':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load the model
    model = YOLO(args.model)
    print(f"Model loaded: {args.model}")
    
    # Train with optimized parameters for maximum accuracy
    print(f"Training with {args.epochs} epochs, image size {args.imgsz}...")
    results = model.train(
        data=args.data, 
        epochs=args.epochs,
        device=device,
        single_cls=args.single_cls, 
        mosaic=args.mosaic,
        optimizer=args.optimizer, 
        lr0=args.lr0, 
        lrf=args.lrf, 
        momentum=args.momentum,
        batch=args.batch,
        imgsz=args.imgsz,
        amp=args.amp,                 # Automatic mixed precision
        cos_lr=args.cos_lr,              # Use cosine learning rate
        warmup_epochs=args.warmup_epochs,         # Longer warmup period
        weight_decay=args.weight_decay,      # L2 regularization
        
        # Loss gains
        box=args.box,                  # Box loss gain
        cls=args.cls,                  # Cls loss gain
        dfl=args.dfl,                  # DFL loss gain
        
        # Advanced augmentations
        close_mosaic=args.close_mosaic,         # Disable mosaic later in training
        mixup=args.mixup,                # Image mixup
        copy_paste=args.copy_paste,           # Copy-paste augmentation
        degrees=args.degrees,             # Random rotation
        translate=args.translate,            # Random translation
        scale=args.scale,                # Random scaling
        shear=args.shear,                # Random shear
        perspective=args.perspective,       # Random perspective
        flipud=args.flipud,               # Random flip up-down
        fliplr=args.fliplr,               # Random flip left-right
        hsv_h=args.hsv_h,              # HSV-Hue augmentation
        hsv_s=args.hsv_s,                # HSV-Saturation augmentation
        hsv_v=args.hsv_v,                # HSV-Value augmentation
        
        # Training efficiency
        cache=args.cache,              # Cache images in RAM
        workers=args.workers,                # Number of workers
        
        # Mask parameters
        overlap_mask=args.overlap_mask,        # Masks should overlap during training
        mask_ratio=args.mask_ratio,             # Mask downsample ratio
        
        # Validation and saving
        patience=args.patience,             # Early stopping patience
        save=True,                # Save checkpoints
        save_period=args.save_period,           # Save checkpoint every X epochs
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        plots=args.plots,               # Generate plots for analysis
        val=args.val,                 # Validate during training
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time/3600:.2f} hours ({elapsed_time/60:.2f} minutes)")
    
    # Path to best model
    best_model_path = os.path.join(args.save_dir, 'weights/best.pt')
    print(f"Best model saved at: {best_model_path}")
    
    # Validate the model
    print("\nValidating model on test dataset...")
    try:
        val_results = model.val(data=args.data, split="test")
        print(f"Validation results:")
        print(f"  mAP@0.5     : {val_results.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {val_results.box.map:.4f}")
        print(f"  Precision   : {val_results.box.p:.4f}")
        print(f"  Recall      : {val_results.box.r:.4f}")
    except Exception as e:
        print(f"Error during validation: {e}")
    
    print("\nTraining complete. Use predict.py to test the model on new images.")
'''
Mixup boost val pred but reduces test pred
Mosaic shouldn't be 1.0  
'''


'''
                   from  n    params  module                                       arguments
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]
 22        [15, 18, 21]  1    751507  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]
Model summary: 225 layers, 3,011,043 parameters, 3,011,027 gradients, 8.2 GFLOPs
'''