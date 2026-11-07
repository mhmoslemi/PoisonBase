run_parts 0-3 pretrain --model ConvNetBN --fraction 0.5 --repeat 2 
run_parts 0-2 control --model ConvNetBN --kind plain 
run_parts 0-3 pretrain --model ResNet20BN --fraction 0.2 --repeat 0 
run_parts 0-3 pretrain --model ResNet20BN --fraction 0.2 --repeat 1 
run_parts 0-3 pretrain --model ResNet20BN --fraction 0.2 --repeat 2 
