run_parts 0-3 pretrain --model ResNet20BN --fraction 0.5 --repeat 0 
run_parts 0-3 pretrain --model ResNet20BN --fraction 0.5 --repeat 1 
run_parts 0-3 pretrain --model ResNet20BN --fraction 0.5 --repeat 2 
run_parts 0-2 control --model ResNet20BN --kind plain 
run_parts 0-0 wo_pretrained 
