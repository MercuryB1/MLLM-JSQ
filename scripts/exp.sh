#!/bin/bash                                                                                                                                                                                       
# JSQ v3 metric ablation with GQA calibration + MMStar evaluation

COMMON="--model Qwen/Qwen2-VL-7B-Instruct --calib_dataset gqa --sparsity_ratio 0.4375 --w_bits 8 --a_bits 8 --search_method none --tasks mmstar"
mkdir -p logs
                                                                                                                                                                                                
# Exp 0: Baseline JSQ v1                                  
CUDA_VISIBLE_DEVICES=0 python main.py $COMMON --pruning_method jsq_v1 \
> logs/exp0_jsq_v1.log 2>&1 &
                                                                                                                                                                                                
# Exp 1: density only
CUDA_VISIBLE_DEVICES=1 python main.py $COMMON --pruning_method jsq_v3 --alpha 0.5 --rho 0 --beta 0 \                                                                                              
> logs/exp1_density.log 2>&1 &                                                                                                                                                                  
                                                                                                                                                                                                
# Exp 2: quant_damage only                                                                                                                                                                        
CUDA_VISIBLE_DEVICES=2 python main.py $COMMON --pruning_method jsq_v3 --alpha 0 --rho 0 --beta 0.5 \                                                                                              
> logs/exp2_quant_damage.log 2>&1 &                     
                                                                                                                                                                                                
# Exp 3: density + sensitivity
CUDA_VISIBLE_DEVICES=3 python main.py $COMMON --pruning_method jsq_v3 --alpha 0.5 --rho 1.0 --beta 0 \                                                                                            
> logs/exp3_density_sens.log 2>&1 &                                                                                                                                                             

# Exp 4: density + quant_damage                                                                                                                                                                   
CUDA_VISIBLE_DEVICES=4 python main.py $COMMON --pruning_method jsq_v3 --alpha 0.5 --rho 0 --beta 0.5 \
> logs/exp4_density_qd.log 2>&1 &                                                                                                                                                               
                                                                                                                                                                                                
# Exp 5: 全开                                                                                                                                                                                     
CUDA_VISIBLE_DEVICES=5 python main.py $COMMON --pruning_method jsq_v3 --alpha 0.5 --rho 1.0 --beta 0.5 \                                                                                          
> logs/exp5_full.log 2>&1 &    

CUDA_VISIBLE_DEVICES=0 python main.py --model Qwen/Qwen2-VL-7B-Instruct --calib_dataset gqa --sparsity_ratio 0.4375 --w_bits 8 --a_bits 8 --search_method none --tasks mmstar --pruning_method jsq_v1   