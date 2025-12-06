# Color-Image-Steganography-Using-Generative-Adversarial-Networks-with-a-Phased-Training-Strategy

# training set: 40,000 color spatial images
# traning order: R->B->G
# for training model
python G7_cov_wise_DDLloss_withW_R.py
python stegan_temp4w1.py
python G7_cov_wise_DDLloss_withW_RB.py
python stegan_temp4w2.py
python G7_cov_wise_ucC1_DDLloss_withW_RBG.py

# for generating stegos
python stegan_cost_2wA1.py
python stegan_cost_2wA2.py
python stegan_cost_2wA3.py