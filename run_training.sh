#!/bin/bash

source_domains=("clipart" "sketch" "clipart" "real" "infograph" "real")
target_domains=("sketch" "painting" "quickdraw" "sketch" "real" "painting")
domain_idxs=(1 2)
subset_idxs=(1 2)
samplings=("final_all_loss_distance_and_ui_weighted_0.5")
loss=("all")

for domain_idx in "${domain_idxs[@]}"; do
  source_domain="${source_domains[$domain_idx]}"
  target_domain="${target_domains[$domain_idx]}"
  for subset_idx in "${subset_idxs[@]}"; do
    for sampling in "${samplings[@]}"; do
      python train.py --cfg_file config/domainnet_50/"${source_domain}"2"${target_domain}".yml -a GMM -d self_ft --dataset domainnet_50 \
      --sampling "$sampling" --subset_idx_argument "$subset_idx" --loss "$loss"
    done
  done
done

#python save_extracted_result.py --domain_idxs "${domain_idxs[@]}" --subset_idxs "${subset_idxs[@]}" --samplings "${samplings[@]}"