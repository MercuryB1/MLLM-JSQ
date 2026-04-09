#!/bin/bash                                                                                                                                                                                       
# JSQ v3 metric ablation with GQA calibration + MMStar evaluation                                                                                                                                 
export CUDA_VISIBLE_DEVICES=3                                                                                                                                                                                                
COMMON="--model Qwen/Qwen2-VL-7B-Instruct --calib_dataset gqa --sparsity_ratio 0.4375 --w_bits 8 --a_bits 8 --search_method none --tasks mmstar"                                                  
                                                                                                                                                                                                
# # Exp 0: Baseline JSQ v1                                                                                                                                                                          
# python main.py $COMMON --pruning_method jsq_v1                                                                                                                                                    

# # # Exp 1: 只开 density (alpha=0.5, rho=0, beta=0)                                                                                                                                                  
# python main.py $COMMON --pruning_method jsq_v3 --alpha 0.5 --rho 0 --beta 0

# # Exp 2: 只开 quant_damage (alpha=0, rho=0, beta=0.5)                                                                                                                                             
# python main.py $COMMON --pruning_method jsq_v3 --alpha 0 --rho 0 --beta 0.5
                                                                                                                                                                                                
# # Exp 3: density + sensitivity (alpha=0.5, rho=2.1, beta=0)                                                                                                                                       
# python main.py $COMMON --pruning_method jsq_v3 --alpha 0.5 --rho 2.1 --beta 0
                                                                                                                                                                                                
# # Exp 4: density + quant_damage (alpha=0.5, rho=0, beta=0.5)                                                                                                                                      
# python main.py $COMMON --pruning_method jsq_v3 --alpha 0.5 --rho 0 --beta 0.5
                                                                                                                                                                                                
# # Exp 5: 全开 JSQ v3 (alpha=0.5, rho=2.1, beta=0.5)                                                                                                                                               
python main.py $COMMON --pruning_method jsq_v3 --alpha 0.5 --rho 2.1 --beta 0.5