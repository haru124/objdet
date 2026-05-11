**VSCODE DEBUGGER**



torchvision gives you "for free":



GeneralizedRCNNTransform — image normalization + batching

AnchorGenerator — multi-scale anchor creation

RPNHead — shared conv + objectness/delta predictors

RegionProposalNetwork — NMS, proposal sampling, loss computation

MultiScaleRoIAlign — feature extraction per proposal

TwoMLPHead — FC layers after RoI pooling

FastRCNNPredictor — final classification + regression heads

BoxCoder — encode/decode box deltas

Matcher — IoU-based GT assignment for anchors/proposals

BalancedPositiveNegativeSampler — 1:3 pos/neg ratio sampling







**What you need to do in VSCode**

Set one breakpoint inside \_train\_one\_epoch in trainer.py, on the line:

pythonloss\_dict = self.model(images, targets)

Pause there. Then in the Debug Console (bottom panel, not Terminal), type expressions directly. You can inspect every single component you listed.



Every component you asked about — what to type in Debug Console

GeneralizedRCNNTransform — normalization + batching

python# The transform config is readable directly

model.transform

model.transform.min\_size

model.transform.max\_size

model.transform.image\_mean

model.transform.image\_std



\# To see what it does to your actual images, run it manually:

transformed, \_ = model.transform(images, targets)

transformed.tensors.shape        # batched tensor after resize+normalize

transformed.image\_sizes          # original sizes before padding

Backbone + FPN — feature maps

python# Run backbone on the transformed images

transformed, \_ = model.transform(images, targets)

features = model.backbone(transformed.tensors)

features.keys()                  # FPN level names: '0','1','2','3','pool'

features\['0'].shape              # P2 shape

features\['1'].shape              # P3 shape

features\['2'].shape              # P4 shape

features\['3'].shape              # P5 shape

features\['pool'].shape           # P6 shape

AnchorGenerator — multi-scale anchors

pythontransformed, \_ = model.transform(images, targets)

features = model.backbone(transformed.tensors)

feature\_list = list(features.values())



anchors = model.rpn.anchor\_generator(transformed, feature\_list)

len(anchors)                     # one tensor per image

anchors\[0].shape                 # \[total\_anchors, 4] for image 0

anchors\[0]\[0]                    # first anchor box \[x1,y1,x2,y2]



\# See anchor config

model.rpn.anchor\_generator.sizes

model.rpn.anchor\_generator.aspect\_ratios

RPNHead — objectness logits + bbox deltas

pythontransformed, \_ = model.transform(images, targets)

features = model.backbone(transformed.tensors)

feature\_list = list(features.values())



objectness, bbox\_deltas = model.rpn.head(feature\_list)

len(objectness)                  # one tensor per FPN level

objectness\[0].shape              # \[B, num\_anchors, H, W] at level 0

bbox\_deltas\[0].shape             # \[B, num\_anchors\*4, H, W] at level 0



\# Check all levels

\[(o.shape, d.shape) for o, d in zip(objectness, bbox\_deltas)]

RegionProposalNetwork — proposals after NMS

pythontransformed, \_ = model.transform(images, targets)

features = model.backbone(transformed.tensors)



\# This runs full RPN: anchor gen + head + NMS

proposals, rpn\_losses = model.rpn(transformed, features, targets)

len(proposals)                   # one tensor per image

proposals\[0].shape               # \[N\_proposals, 4] for image 0

rpn\_losses                       # {'loss\_objectness': ..., 'loss\_rpn\_box\_reg': ...}

BoxCoder — encode/decode deltas

pythonmodel.rpn.box\_coder              # see the coder object

model.rpn.box\_coder.weights      # (1.0, 1.0, 1.0, 1.0) default



\# Decode a proposal back to box

model.rpn.box\_coder.decode(bbox\_deltas\[0].reshape(-1,4)\[:5], anchors\[0]\[:5])

Matcher — IoU assignment

pythonmodel.rpn.proposal\_matcher       # RPN matcher

model.rpn.proposal\_matcher.high\_threshold    # 0.7

model.rpn.proposal\_matcher.low\_threshold     # 0.3



model.roi\_heads.proposal\_matcher             # ROI matcher  

model.roi\_heads.proposal\_matcher.high\_threshold   # 0.5

model.roi\_heads.proposal\_matcher.low\_threshold    # 0.5

BalancedPositiveNegativeSampler

pythonmodel.rpn.fg\_bg\_sampler          # RPN sampler

model.rpn.fg\_bg\_sampler.batch\_size\_per\_image   # 256

model.rpn.fg\_bg\_sampler.positive\_fraction      # 0.5



model.roi\_heads.fg\_bg\_sampler    # ROI sampler

model.roi\_heads.fg\_bg\_sampler.batch\_size\_per\_image   # 512

model.roi\_heads.fg\_bg\_sampler.positive\_fraction      # 0.25

MultiScaleRoIAlign — ROI pooling

pythontransformed, \_ = model.transform(images, targets)

features = model.backbone(transformed.tensors)

proposals, \_ = model.rpn(transformed, features, targets)



\# Run ROI align manually

box\_features = model.roi\_heads.box\_roi\_pool(

&#x20;   features, proposals, transformed.image\_sizes

)

box\_features.shape               # \[total\_proposals, 256, 7, 7]



\# See config

model.roi\_heads.box\_roi\_pool.output\_size    # (7, 7)

model.roi\_heads.box\_roi\_pool.sampling\_ratio # 2

model.roi\_heads.box\_roi\_pool.featmap\_names  # \['0','1','2','3']

TwoMLPHead — FC layers

pythonbox\_features\_flat = model.roi\_heads.box\_head(box\_features)

box\_features\_flat.shape          # \[total\_proposals, 1024]



\# See the layers

model.roi\_heads.box\_head.fc6     # Linear(12544, 1024)

model.roi\_heads.box\_head.fc7     # Linear(1024, 1024)

FastRCNNPredictor — final classification + regression

pythonclass\_logits, box\_regression = model.roi\_heads.box\_predictor(box\_features\_flat)

class\_logits.shape               # \[total\_proposals, num\_classes]

box\_regression.shape             # \[total\_proposals, num\_classes\*4]



model.roi\_heads.box\_predictor.cls\_score   # Linear(1024, 9)

model.roi\_heads.box\_predictor.bbox\_pred   # Linear(1024, 36)



How to set this up in VSCode

.vscode/launch.json

json{

&#x20;   "version": "0.2.0",

&#x20;   "configurations": \[

&#x20;       {

&#x20;           "name": "Train (debug mode)",

&#x20;           "type": "debugpy",

&#x20;           "request": "launch",

&#x20;           "program": "${workspaceFolder}/main.py",

&#x20;           "args": \[

&#x20;               "--mode", "train",

&#x20;               "--exp", "config/experiments/exp\_01\_smoke\_test.yaml"

&#x20;           ],

&#x20;           "console": "integratedTerminal",

&#x20;           "justMyCode": false

&#x20;       }

&#x20;   ]

}

Set justMyCode: false so you can step into torchvision source files and see exactly what happens inside model.rpn.forward(), model.roi\_heads.forward(), etc.



The conclusion

What you want to seeHow to see itTensor shapes at every stageBreakpoint + Debug Console expressions aboveStep inside torchvision RPN codeF11 with justMyCode:falseWatch a variable change across iterationsAdd to Watch panelSee anchor values, proposal coordsExpand tensor in Variables panelCheck loss values per componentHover over loss\_dictInspect matcher thresholdsType model.rpn.proposal\_matcher.high\_threshold in console

Delete all debug functions and the three stub files (fpn.py, rpn.py, roi\_heads.py). The VSCode debugger gives you more information, interactively, with zero maintenance cost. The only code change needed is the launch.json above.

