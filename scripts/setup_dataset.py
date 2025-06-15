# Setup dataset, unzip, organize folders, and create data.yaml
import os
yaml_text = '''
path: /tmp/dataset
train: images/train
val: images/val
test: images/test
names:
  0: Person
  1: Car
  2: Bicycle
  3: OtherVehicle
  4: DontCare
nc: 5
'''
with open("data.yaml", "w") as f:
    f.write(yaml_text)
