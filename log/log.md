# New record

---

Acc : 78.206
F1 : 0.745

- mask and gender

eff b0, lr=0.00006, bs=64, randaug+cutout, inputsize=224, loss_fn=LS(0.05), WeightedSampler, age >=58
epoch 21)
train Loss: 0.3310 Acc: 0.9638 F1: 0.963728513759618
valid Loss: 0.3071 Acc: 0.9786 F1: 0.9791634664746235

- age

eff b0

extended_far_best

---

Acc : 79.079
F1 : 0.751

- mask and gender

model=eff b0, optimizer=Adam, lr=0.00006, bs=64, augment=randaug+cutout, inputsize=224, loss_fn=LS(0.05), split 20%
    * epoch 27)
      - train Loss: 0.3211 Acc: 0.9668 F1: 0.9668728849468181
      - valid Loss: 0.2878 Acc: 0.9854 F1: 0.9842203787970396

- age

eff b0
extended_far_best